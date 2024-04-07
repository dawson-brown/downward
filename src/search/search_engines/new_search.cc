#include "new_search.h"

#include "../evaluation_context.h"
#include "../evaluator.h"
#include "../open_list_factory.h"
#include "../pruning_method.h"

#include "../algorithms/ordered_set.h"
#include "../plugins/options.h"
#include "../task_utils/successor_generator.h"
#include "../utils/rng_options.h"
#include "../utils/logging.h"

#include <cassert>
#include <cstdlib>
#include <memory>
#include <optional.hh>
#include <set>
#include <random>



#include <chrono>

using namespace std;

namespace new_search {
NewSearch::NewSearch(const plugins::Options &opts)
    : SearchEngine(opts),
      reopen_closed_nodes(opts.get<bool>("reopen_closed")),
      evaluator(opts.get<shared_ptr<Evaluator>>("eval")),
      rng(utils::parse_rng_from_options(opts)),
      search_tree(SparseSearchTree(0.0, opts.get<double>("tau"), rng)) {}

void NewSearch::initialize() {
    log << "Conducting new search"
        << (reopen_closed_nodes ? " with" : " without")
        << " reopening closed nodes, (real) bound = " << bound
        << endl;

    State initial_state = state_registry.get_initial_state();

    EvaluationContext eval_context(initial_state, 0, false, &statistics);
    int init_eval = eval_context.get_evaluator_value_or_infinity(evaluator.get());
    if (init_eval != numeric_limits<int>::max()) {
        statistics.inc_evaluated_states();

        if (search_progress.check_progress(eval_context))
            statistics.print_checkpoint_line(0);
        start_f_value_statistics(eval_context);
        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();
        search_tree.add_initial_state(initial_state.get_id(), init_eval);

        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(initial_state, applicable_ops);
        // known_states[initial_state] = KnownState(0, 0, TODO, applicable_ops);

    }

    print_initial_evaluator_values(eval_context);

    // pruning_method->initialize(task);
}

void NewSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
}

// StateID NewSearch::select_next_state(StateLoc& loc) {
//     int selected_depth = type_buckets.begin()->first;
//     double r = rng->random();
    
//     double total_sum = current_sum;
//     double p_sum = 0.0;
//     for (auto it : type_buckets) {
//         double p = 1.0 / total_sum;
//         p *= std::exp(-1.0*static_cast<double>(it.first) / tau); //remove -1.0 *
//         p *= static_cast<double>(it.second.size());
//         p_sum += p;
//         if (r <= p_sum) {
//             selected_depth = it.first;
//             break;
//         }
//     }

//     vector<vector<StateID>> &types = type_buckets.at(selected_depth);
//     int chosen_type_i = rng->random(types.size());
//     vector<StateID> &states = types[chosen_type_i];
//     int chosen_i = rng->random(states.size());
//     StateID id = states[chosen_i];

//     loc.depth = selected_depth;
//     loc.breadth = chosen_type_i;
//     loc.i = chosen_i;
//     return id;
// }

// void NewSearch::remove_from_type_buckets(StateLoc loc) {
//     vector<vector<StateID>> &types = type_buckets.at(loc.depth);
//     vector<StateID> &states = types[loc.breadth];

//     utils::swap_and_pop_from_vector(states, loc.i);
//     if (states.empty()){
//         utils::swap_and_pop_from_vector(types, loc.breadth);
//         if (types.empty()) {
//             type_buckets.erase(loc.depth);
//             current_sum -= std::exp(-1.0*static_cast<double>(loc.depth) / tau);
//         }
//     }
// }

// void NewSearch::insert_into_type_buckets(StateID s_id, StateLoc& loc) {
//     vector<vector<StateID>>& types = type_buckets[loc.depth];
//     if (loc.breadth == -1) {
//         types.push_back({s_id});
//         current_sum += std::exp(static_cast<double>(loc.depth) / tau);

        
//         loc.breadth = types.size()-1;
//         loc.i = types[loc.breadth].size()-1;
//     } else {
//         types[loc.breadth].push_back(s_id);
//         loc.i = types[loc.breadth].size()-1;
//     }
// }

// OperatorID NewSearch::get_next_rollout_start(const State s) {

//     int next_op_id = known_states[s].op_index++;
//     if (next_op_id == known_states[s].operators.size() - 1) { // every successor has been rolled out from
//         SearchNode node = search_space.get_node(s);
//         node.close();
//         known_states[s].status |= RAND_EXPAND;
//     }
//     return known_states[s].operators[next_op_id];
    
// }

OperatorID NewSearch::random_next_action(const State s) {
    // return *rng->choose(known_states[s].operators);
}

bool NewSearch::add_state(const StateID s_id, const OperatorID op_id, const StateID parent_s_id) {
    // State s = state_registry.lookup_state(s_id);
    // State parent_s = state_registry.lookup_state(parent_s_id);

    // SearchNode to_open = search_space.get_node(s);
    // vector<OperatorID> applicable_ops;
    // successor_generator.generate_applicable_ops(s, applicable_ops);
    // if (applicable_ops.size() == 0) { // dont add deadends
    //     return false;
    // }

    // SearchNode parent = search_space.get_node(parent_s);
    // OperatorProxy op = task_proxy.get_operators()[op_id];
    // int cost = get_adjusted_cost(op);

    // if (to_open.is_new()) {

    //     rng->shuffle(applicable_ops); // randomize successor order

    //     if (loc.breadth == -1) {
    //         known_states[s] = KnownState(0, 0, TODO, applicable_ops); // new type gets depth of 0 in type
    //     } else {
    //         known_states[s] = KnownState(known_states[parent_s].depth + 1, 0, TODO, applicable_ops);
    //     }

    //     to_open.open(parent, op, cost);
    //     insert_into_type_buckets(s_id, loc);
    //     return true;

    // } else if (to_open.get_g() > parent.get_g() + cost) { // open or closed, update g-cost and depth
    //     to_open.update_parent(parent, op, cost);
    // }
    // return true;
}

// do a greedy rollout until you hit a goal or the bottom of a crater.
std::pair<NewSearch::RolloutResult, StateID> NewSearch::greedy_rollout(const State rollout_state) {

    SearchNode rollout_node = search_space.get_node(rollout_state);
    int rollout_g = rollout_node.get_g();
    State curr_state = rollout_state;
    State next_state = curr_state;

    EvaluationContext eval_context(curr_state, &statistics, false);
    int curr_h = eval_context.get_evaluator_value_or_infinity(evaluator.get());
    while (true) { // keep expanding while heuristic improvements come in

        int min_child_h = std::numeric_limits<int>::max();
        OperatorID chosen_op_id = OperatorID::no_operator;
        int chosen_op_index = 0;
        int index = 0;

        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(curr_state, applicable_ops);
        if (applicable_ops.size() == 0) {
            return make_pair(DEADEND, curr_state.get_id());
        }

        for (OperatorID op_id : applicable_ops) {
            OperatorProxy op = task_proxy.get_operators()[op_id];
            // if ((node->get_real_g() + op.get_cost()) >= bound)
            //     continue;

            State succ_state = state_registry.get_successor_state(curr_state, op);
            statistics.inc_generated();
            EvaluationContext succ_eval_context(
                    succ_state, &statistics, false);

            int eval = succ_eval_context.get_evaluator_value_or_infinity(evaluator.get());

            if (eval < min_child_h) {
                chosen_op_index = index;
                chosen_op_id = op_id;
                min_child_h = eval;
                next_state = succ_state;
            }
            index+=1;

        }

        if (task_properties::is_goal_state(task_proxy, next_state)) {
            Plan plan;
            search_space.trace_path(next_state, plan);
            set_plan(plan);
            return make_pair(GOAL, next_state.get_id()); // end of recursion: success
        }

        if (min_child_h >= curr_h) {
            break;
        }

        curr_h = min_child_h;
        curr_state = next_state;
    }
    
    return make_pair(HI, curr_state.get_id()); // bottom of crater
     
}


std::pair<NewSearch::RolloutResult, StateID> NewSearch::random_rollout() {

    RolloutCTX next_rollout = search_tree.select_next_state();
    State parent_s = state_registry.lookup_state(next_rollout.state_id);
    State curr_state = parent_s;

    int rollout_limit = next_rollout.r_length;
    do {
        OperatorID op_id = random_next_action(curr_state);
        OperatorProxy op = task_proxy.get_operators()[op_id];
        curr_state = state_registry.get_successor_state(parent_s, op);

        if (task_properties::is_goal_state(task_proxy, curr_state)) {
            return make_pair(GOAL, curr_state.get_id()); // end of recursion: success
        }

        rollout_limit-=1;
    } while(rollout_limit > 0);

    SearchNode curr_node = search_space.get_node(curr_state);
    EvaluationContext succ_eval_context(
            curr_state, &statistics, false);
    int eval = succ_eval_context.get_evaluator_value_or_infinity(evaluator.get());
    
    if (eval < next_rollout.h) { // if the rollout found an improvement
        return greedy_rollout(curr_state);

    } else {
        return make_pair(UHR, curr_state.get_id());
    }

}

SearchStatus NewSearch::step() {
    
    pair<RolloutResult, StateID> result_pair = random_rollout();
    RolloutResult result = result_pair.first;
    StateID state_id = result_pair.second;
    if (result == GOAL) {

        return SOLVED;
    } else if (result == DEADEND) {
        
        return IN_PROGRESS;
    } else if (result == HI) {

        return IN_PROGRESS;
    } else {

        return IN_PROGRESS;
    }

    // State expanding_state = node->get_state();
    // int h;
    // if (node->get_info().creating_operator == OperatorID::no_operator) {
    //     EvaluationContext succ_eval_context(
    //         expanding_state, &statistics, false);
    //     h = succ_eval_context.get_evaluator_value_or_infinity(evaluator.get());
    // } else {
    //     OperatorProxy op = task_proxy.get_operators()[node->get_info().creating_operator];
    //     int succ_g = node->get_g() + get_adjusted_cost(op);
    //     EvaluationContext succ_eval_context(
    //         expanding_state, succ_g, false, &statistics);
    //     h = succ_eval_context.get_evaluator_value_or_infinity(evaluator.get());
    // }
}

void NewSearch::reward_progress() {
    // Boost the "preferred operator" open lists somewhat whenever
    // one of the heuristics finds a state with a new best h value.
    // open_list->boost_preferred();
}

void NewSearch::dump_search_space() const {
    search_space.dump(task_proxy);
}

void NewSearch::start_f_value_statistics(EvaluationContext &eval_context) {
    // if (f_evaluator) {
    //     int f_value = eval_context.get_evaluator_value(f_evaluator.get());
    //     statistics.report_f_value_progress(f_value);
    // }
}

/* TODO: HACK! This is very inefficient for simply looking up an h value.
   Also, if h values are not saved it would recompute h for each and every state. */
void NewSearch::update_f_value_statistics(EvaluationContext &eval_context) {
    // if (f_evaluator) {
    //     int f_value = eval_context.get_evaluator_value(f_evaluator.get());
    //     statistics.report_f_value_progress(f_value);
    // }
}

void add_options_to_feature(plugins::Feature &feature) {
    // SearchEngine::add_pruning_option(feature);
    SearchEngine::add_options_to_feature(feature);
}
}
