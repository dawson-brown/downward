#include "sparse_mcts.h"

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

namespace sparse_mcts_search {
SparseMCTS::SparseMCTS(const plugins::Options &opts)
    : SearchAlgorithm(opts),
      heuristic(opts.get<shared_ptr<Evaluator>>("eval", nullptr)),
      rng(utils::parse_rng_from_options(opts)),
      c((float)opts.get<double>("c")),
      epsilon((float)opts.get<double>("epsilon")),
      theta((float)opts.get<double>("theta")) {}



bool SparseMCTS::is_dead_end(
    EvaluationContext &eval_context) const {
    return eval_context.is_evaluator_value_infinite(heuristic.get());
}

void SparseMCTS::initialize() {
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

        // shared_ptr<Node> node(new Node(
        //     initial_state.get_id(),
        //     OperatorID::no_operator,
        //     nullptr,
        //     eval_context.get_evaluator_value(heuristic.get())
        // ));
        root.reset(new Node(
            initial_state.get_id(),
            OperatorID::no_operator,
            nullptr,
            eval_context.get_evaluator_value(heuristic.get())
        ));
        seen_states[initial_state] = root;
        cached_select = root;

    }

    print_initial_evaluator_values(eval_context);

}

void SparseMCTS::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
}

shared_ptr<SparseMCTS::Node> SparseMCTS::select(shared_ptr<SparseMCTS::Node> node)
{
    if (rng->random() < pow(theta, node->num_visits)) {
        return node;
    }

    shared_ptr<Node> best = node;
    float bestScore = numeric_limits<float>::min();
    auto& children = best->children;
    // UCT
    for (auto& n : children) {
        float score = n->utc(c, root->num_visits);
        if (score > bestScore) {
            bestScore = score;
            best = n;
        }
    }

    return best;
}

SparseMCTS::Outcome SparseMCTS::expand(SparseMCTS::Node &node, vector<OperatorID> &path) {
    StateID start_id = node.id;
    Outcome oc;
    int h_to_beat = node.h;
    OperatorID best_op_id = OperatorID::no_operator;

    State parent_s = state_registry.lookup_state(start_id);
    StateRegistry tmp_registry(task_proxy);
    State curr_state = parent_s;
    vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(parent_s, applicable_ops);

    for(OperatorID op_id : applicable_ops) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        curr_state = tmp_registry.get_successor_state(parent_s, op);
        
        if (task_properties::is_goal_state(task_proxy, curr_state)) {
            cout << "!!! Goal found: random_rollout !!!" << endl;
            path.push_back(op_id);

            // for (auto op_id : curr_rollout_path) {
            //     cout << task_proxy.get_operators()[op_id].get_name() << endl << endl;
            // }
            return Outcome(GOAL, 0); // end of recursion: success
        }

        EvaluationContext succ_eval_context(
            curr_state, &statistics, false);
        statistics.inc_evaluated_states();
        int eval = succ_eval_context.get_evaluator_value_or_infinity(heuristic.get());

        if (eval < h_to_beat) {
            h_to_beat = eval;
            best_op_id = op_id;
        }
    }

    // node.rollout_step = applicable_ops.size()-1; // maybe
    if (h_to_beat == node.h) {
        return Outcome(UHR, h_to_beat);
    } else {
        path.push_back(best_op_id);
        return Outcome(HI, h_to_beat);
    }
}

SparseMCTS::Outcome SparseMCTS::simulate(SparseMCTS::Node &node, vector<OperatorID> &path) {
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
            cout << "!!! Goal found: random_rollout !!!" << endl;

            // for (auto op_id : curr_rollout_path) {
            //     cout << task_proxy.get_operators()[op_id].get_name() << endl << endl;
            // }
            return Outcome(GOAL, 0); // end of recursion: success
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
        return Outcome(UHR, eval);

    }

}

void SparseMCTS::back_propogate(Result result, SparseMCTS::Node &node) {

    float score = 1 ? result == HI : 0;
    node.update(score);
    
    if (result == HI) {
        node.rollout_step = node.rl_length;
    } else {
        node.rollout_step+=1;
    }

    std::shared_ptr<Node> current = node.parent;
    while (current) {
        current->update(score);
        current = current->parent;
    }
}

SearchStatus SparseMCTS::step()
{
    shared_ptr<Node> selected = root;
    bool greedy = false;
    if (cached_select != nullptr) {
        selected = cached_select;
        greedy = true;
    }
    else {
        while (true) {
            shared_ptr<Node> tmp = select(selected);

            if (tmp == selected)
                break;
            selected = tmp;
        }
    }

    vector<OperatorID> rollout_path;
    Outcome oc;
    if (greedy) {
        oc = expand(*selected, rollout_path);
        statistics.inc_expanded();
    } else {
        oc = simulate(*selected, rollout_path);
    }

    if (oc.result == GOAL) {
        shared_ptr<Node> goal_node = open_path_to_new_node(selected, rollout_path, oc, false);
        State goal = state_registry.lookup_state(goal_node->id);
        if (check_goal_and_set_plan(goal))
            return SOLVED;
    }

    if (oc.result != DEADEND) {
         if (oc.result == HI) {
            cached_select = open_path_to_new_node(selected, rollout_path, oc, true ? greedy : false);
            statistics.inc_generated();
            statistics.print_checkpoint_line(rollout_path.size());
            back_propogate(oc.result, *selected);
         } else {
            cached_select = nullptr;
            if (rng->random() < epsilon && !greedy) {
                cached_select = open_path_to_new_node(selected, rollout_path, oc, false);
                statistics.inc_generated();
                statistics.print_checkpoint_line(rollout_path.size());
            }
            back_propogate(oc.result, *selected);
        }
    }
    return IN_PROGRESS;
}

shared_ptr<SparseMCTS::Node> SparseMCTS::open_path_to_new_node(
    shared_ptr<SparseMCTS::Node> selected, 
    std::vector<OperatorID> path, 
    Outcome oc, bool bump) 
{
    State state = state_registry.lookup_state(selected->id);
    OperatorID last_op(OperatorID::no_operator);

    int i = 0;
    bool close_behind = false;
    bool last_was_new = false;
    for (OperatorID op_id : path) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        State succ_state = state_registry.get_successor_state(state, op);

        SearchNode node = search_space.get_node(state);
        SearchNode succ_node = search_space.get_node(succ_state);

        if (i>0 && close_behind) {
            node.close(); // close behind you so only type states are open
        } else {
            close_behind = true;
        }

        if (succ_node.is_new()) {
            succ_node.open(node, op, get_adjusted_cost(op));
            last_was_new = true; 
        } else if (succ_node.get_g() > node.get_g() + get_adjusted_cost(op)) {
            succ_node.update_parent(node, op, get_adjusted_cost(op));
            if (succ_node.is_open()) {
                close_behind = false; // don't close states that were already open (those states are or were in the type system)
            }
            last_was_new = false;
        }

        state = succ_state;
        last_op = op_id;

        i++;
    }

    if (bump) {
        selected->id = state.get_id();
        selected->op_id = last_op;
        selected->h = oc.h;
        return selected;
    } else {
        if (last_was_new) { // if the last state of the path was new, make a new node and include it in seen_states
            shared_ptr<Node> node(new Node(
                state.get_id(),
                last_op,
                selected,
                oc.h
            ));
            selected->add_child(node);
            seen_states[state] = node;
            return node;
        } else {
            return seen_states[state];
        }
    }
}

// void SparseMCTS::add_mcts_node(Node& selected, shared_ptr<Node> new_node) {

// }

void SparseMCTS::dump_search_space() const {
    search_space.dump(task_proxy);
}
}
