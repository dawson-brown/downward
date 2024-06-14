#include "sparse_lwm.h"

#include "../evaluation_context.h"
#include "../evaluator.h"
#include "../open_list_factory.h"
#include "../pruning_method.h"

#include "../task_utils/task_properties.h"
#include "../algorithms/ordered_set.h"
#include "../plugins/options.h"
#include "../task_utils/successor_generator.h"
#include "../utils/logging.h"
#include "../utils/rng_options.h"

#include <cassert>
#include <cstdlib>
#include <memory>
#include <optional>
#include <set>
#include <math.h>

using namespace std;

namespace sparse_lwm_search {
SparseLWM::SparseLWM(const plugins::Options &opts)
    : SearchAlgorithm(opts),
      heuristic(opts.get<shared_ptr<Evaluator>>("eval", nullptr)),
      rng(utils::parse_rng_from_options(opts)),
      tau((float)opts.get<double>("tau")),
      current_sum(0.0) {}



bool SparseLWM::is_dead_end(
    EvaluationContext &eval_context) const {
    return eval_context.is_evaluator_value_infinite(heuristic.get());
}

void SparseLWM::initialize() {
    log << "Conducting Sparse Monte Carlo Tree Search"
        << ", (real) bound = " << bound
        << endl;

    State initial_state = state_registry.get_initial_state();

    /*
      Note: we consider the initial state as reached by a preferred
      operator.
    */
    EvaluationContext eval_context(initial_state, 0, true, &statistics);
    statistics.inc_evaluated_states();

    if (is_dead_end(eval_context)) {
        log << "Initial state is a dead end." << endl;
    } else {
        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();
        depth_buckets[0].push_back({ Node(
            initial_state.get_id(),
            eval_context.get_evaluator_value(heuristic.get())
        )} );
    }
    cached_select = NodeLoc(0, 0, 0);

    print_initial_evaluator_values(eval_context);

}

void SparseLWM::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
}

SparseLWM::NodeLoc SparseLWM::select()
{
   int selected_d = depth_buckets.begin()->first;
    if (depth_buckets.size() > 1) {
        double r = rng->random();
        
        double total_sum = current_sum;
        double p_sum = 0.0;
        for (auto it : depth_buckets) {
            double p = 1.0 / total_sum;
            p *= std::exp(static_cast<double>(it.first) / tau); //remove -1.0 *
            p *= static_cast<double>(it.second.size());
            p_sum += p;
            if (r <= p_sum) {
                selected_d = it.first;
                break;
            }
            // count_i+=1; 
        }
        // cout << p_sum << endl;
    }
    // cout << "depth: " << selected_d << endl;


    vector<vector<Node>> &buckets = depth_buckets[selected_d];
    assert(!buckets.empty());
    int bucket_i = rng->random(buckets.size());
    vector<Node>& bucket = buckets[bucket_i];
    int i = rng->random(bucket.size());

    return NodeLoc(selected_d, bucket_i, i);
}

SparseLWM::Outcome SparseLWM::exploit(SparseLWM::Node &node, vector<OperatorID> &path) {
    Outcome oc;
    int h_to_beat = node.h;

    State curr_state = state_registry.lookup_state(node.id);
    State succ_state = curr_state;
    StateRegistry tmp_registry(task_proxy);

    do {
        int best_h = h_to_beat;
        OperatorID best_op_id = OperatorID::no_operator;
        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(curr_state, applicable_ops);

        if (applicable_ops.size() == 0) {
            return Outcome(DEADEND, numeric_limits<int>::max());
        }

        for(OperatorID op_id : applicable_ops) {
            OperatorProxy op = task_proxy.get_operators()[op_id];
            State tmp_state = tmp_registry.get_successor_state(curr_state, op);
            
            if (task_properties::is_goal_state(task_proxy, tmp_state)) {
                cout << "!!! Goal found: greedy rollout !!!" << endl;
                path.push_back(op_id);
                return Outcome(GOAL, 0); 
            }

            EvaluationContext succ_eval_context(
                tmp_state, &statistics, false);
            statistics.inc_evaluated_states();
            int eval = succ_eval_context.get_evaluator_value_or_infinity(heuristic.get());
            if (eval < h_to_beat) {
                best_h = eval;
                best_op_id = op_id;
                succ_state = tmp_state;
            }
        }

        if (best_h < h_to_beat) {
            path.push_back(best_op_id);
            h_to_beat = best_h;
            curr_state = succ_state;
        } else {
            return Outcome(CRATER, h_to_beat); // if the first expansion fails, the path will be empty
        }
    } while(true);
}

SparseLWM::Outcome SparseLWM::explore(SparseLWM::Node &node, vector<OperatorID> &path) {
    StateID start_id = node.id;
    Outcome oc;

    State parent_s = state_registry.lookup_state(start_id);
    StateRegistry rollout_registry(task_proxy);
    State curr_state = parent_s;
    vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(parent_s, applicable_ops);

    node.set_rollout_length();
    int rollout_limit = node.rl_length;
    do {
        OperatorID op_id = *rng->choose(applicable_ops);
        OperatorProxy op = task_proxy.get_operators()[op_id];
        curr_state = rollout_registry.get_successor_state(parent_s, op);
        
        applicable_ops.clear();
        successor_generator.generate_applicable_ops(curr_state, applicable_ops);
        if (applicable_ops.size() == 0) {
            return Outcome(DEADEND);
        }

        path.push_back(op_id);
        if (task_properties::is_goal_state(task_proxy, curr_state)) {
            cout << "!!! Goal found: random rollout !!!" << endl;
            return Outcome(GOAL, 0);
        }
        parent_s = curr_state;

        rollout_limit-=1;
    } while(rollout_limit > 0);

    EvaluationContext succ_eval_context(
            curr_state, &statistics, false);
    statistics.inc_evaluated_states();
    int eval = succ_eval_context.get_evaluator_value_or_infinity(heuristic.get());
    if (eval == EvaluationResult::INFTY) {
        return Outcome(DEADEND);
    }

    if (eval < node.h) { // if the rollout found an improvement
        // return make_pair(HI, curr_rollout_path);
        return Outcome(HI, eval);

    } else {
        return Outcome(PLATEAU, eval);

    }

}

SearchStatus SparseLWM::step()
{
    NodeLoc selected_loc;
    vector<OperatorID> rollout_path;
    Outcome oc;
    if (cached_select.depth != -1) {
        // cout << "WHY" << endl;
        selected_loc = cached_select;
        Node selected = get_node(selected_loc);
        oc = exploit(selected, rollout_path);
        statistics.inc_expanded();
    }
    else {
        // cout << "HERE" << endl;
        selected_loc = select();
        Node selected = get_node(selected_loc);
        oc = explore(selected, rollout_path);
    }

    if (oc.result == GOAL) {
        Node goal_node = trace_path_to_new_node(selected_loc, rollout_path, oc);
        State goal = state_registry.lookup_state(goal_node.id);
        if (check_goal_and_set_plan(goal))
            return SOLVED;
    }

    if (oc.result != DEADEND) {
        Node new_node = trace_path_to_new_node(selected_loc, rollout_path, oc);
        NodeLoc added_loc = add_node_to_type_system(new_node, selected_loc, oc);
        statistics.inc_generated();
        if (oc.result == HI) {
            cached_select = added_loc; //NodeLoc(selected_loc.depth+1, depth_buckets.at(selected_loc.depth+1).size()-1); // cringe
            statistics.print_checkpoint_line(rollout_path.size());
        } else {
            cached_select = NodeLoc(-1, -1, -1);
        }
    }
    return IN_PROGRESS;
}

// Node pop_selected = utils::swap_and_pop_from_vector(depth_buckets.at(selected_loc.depth), selected_loc.i);
//     if (depth_buckets.at(selected_loc.depth).empty()) {
//         depth_buckets.erase(selected_loc.depth);
//     }
//     current_sum -= std::exp(static_cast<double>(selected_loc.depth) / tau);

SparseLWM::NodeLoc SparseLWM::add_node_to_type_system(SparseLWM::Node new_node, NodeLoc parent_loc, Outcome oc){

    if (oc.result == HI || oc.result == CRATER) {
        int new_depth = parent_loc.depth+1;
        current_sum += std::exp(static_cast<double>(new_depth) / tau);
        depth_buckets[new_depth].push_back({new_node});
        return NodeLoc(new_depth, depth_buckets[new_depth].size()-1, 0);
    } else {
        depth_buckets[parent_loc.depth][parent_loc.bucket_i].push_back({new_node});
        return NodeLoc(parent_loc.depth, parent_loc.bucket_i, depth_buckets[parent_loc.depth][parent_loc.bucket_i].size()-1);
    }

    // if (last_was_new) { // if the last state of the path was new, make a new node and include it in seen_states
    //     Node node(state.get_id(), oc.h);

    //     int new_depth = oc.result == HI ? selected_loc.depth+1 : selected_loc.depth;
    //     current_sum += std::exp(static_cast<double>(new_depth) / tau);
    //     depth_buckets[new_depth].push_back(node);

    //     seen_states[state] = node;
    //     return node;
    // } else {
    //     return seen_states[state];
    // }
}

SparseLWM::Node SparseLWM::trace_path_to_new_node(
    SparseLWM::NodeLoc selected_loc, 
    vector<OperatorID> &path, 
    Outcome oc) 
{
    Node selected = depth_buckets[selected_loc.depth][selected_loc.bucket_i][selected_loc.node_i];
    State state = state_registry.lookup_state(selected.id);

    int i = 0;
    for (OperatorID op_id : path) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        State succ_state = state_registry.get_successor_state(state, op);

        SearchNode node = search_space.get_node(state);
        SearchNode succ_node = search_space.get_node(succ_state);

        if (succ_node.is_new()) {
            succ_node.open(node, op, get_adjusted_cost(op));

            if (i<path.size()-1) {
                succ_node.close(); 
            }
        }
/*
        else if (succ_node.get_g() > node.get_g() + get_adjusted_cost(op)) {
            succ_node.update_parent(node, op, get_adjusted_cost(op));
            if (succ_node.is_open()) {
                close_behind = false; // don't close states that were already open (those states are or were in the type system)
            }
        }
*/
        state = succ_state;
        i++;
    }

    return Node(state.get_id(), oc.h);
}

// void SparseLWM::add_mcts_node(Node& selected, shared_ptr<Node> new_node) {

// }

void SparseLWM::dump_search_space() const {
    search_space.dump(task_proxy);
}
}
