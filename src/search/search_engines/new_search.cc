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
      epsilon(opts.get<double>("epsilon")),
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

    }

    print_initial_evaluator_values(eval_context);
}

void NewSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
}


void NewSearch::open_states_along_rollout_path(RolloutCTX ctx, vector<NewSearch::RolloutNode> &path) {
    State parent_state = state_registry.lookup_state(ctx.state_id);

    for (NewSearch::RolloutNode r_node : path) {
        State succ_state = state_registry.lookup_state(r_node.state_id);
        SearchNode succ_node = search_space.get_node(succ_state);
        OperatorProxy op = task_proxy.get_operators()[r_node.op_id];

        SearchNode parent_node = search_space.get_node(parent_state);
        if (succ_node.is_new()) {
            statistics.inc_evaluated_states();
            succ_node.open(parent_node, op, get_adjusted_cost(op));
        } else {
           if (succ_node.get_g() > parent_node.get_g() + get_adjusted_cost(op)) {
                succ_node.update_parent(parent_node, op, get_adjusted_cost(op));
           }
       }
       parent_state = succ_state;
    }
}


// do a greedy rollout until you hit a goal or the bottom of a hill.
NewSearch::RolloutResult NewSearch::greedy_rollout(const State rollout_state, vector<RolloutNode> &path_so_far) {

    SearchNode rollout_node = search_space.get_node(rollout_state);
    State curr_state = rollout_state;
    State next_state = curr_state;

    int curr_h = path_so_far.back().h;
    while (true) { // keep expanding while heuristic improvements come in

        int min_child_h = curr_h;
        OperatorID chosen_op_id = OperatorID::no_operator;
        int chosen_op_index = 0;
        int index = 0;

        vector<OperatorID> applicable_ops;
        successor_generator.generate_applicable_ops(curr_state, applicable_ops);
        if (applicable_ops.size() == 0) {
            return DEADEND;
        }

        RolloutNode next_rollout_node;
        for (OperatorID op_id : applicable_ops) { // get min-h successor
            OperatorProxy op = task_proxy.get_operators()[op_id];

            State succ_state = state_registry.get_successor_state(curr_state, op);
            statistics.inc_generated();
            
            if (task_properties::is_goal_state(task_proxy, succ_state)) {
                path_so_far.push_back(RolloutNode(succ_state.get_id(), op_id));
                return GOAL; // end of recursion: success
            }

            EvaluationContext succ_eval_context(
                    succ_state, &statistics, false);
            int eval = succ_eval_context.get_evaluator_value_or_infinity(evaluator.get());

            if (eval < min_child_h) {
                min_child_h = eval;
                next_state = succ_state;
                next_rollout_node.op_id = op_id;
                next_rollout_node.state_id = next_state.get_id();
                next_rollout_node.h = eval;
            }
            index+=1;

        }

        if (next_rollout_node.state_id == StateID::no_state) { // bottom of crater
            break;
        }

        path_so_far.push_back(next_rollout_node);
        curr_h = min_child_h;
        curr_state = next_state;
    }
    
    return HI; // bottom of crater
     
}


pair<NewSearch::RolloutResult, vector<NewSearch::RolloutNode>> NewSearch::random_rollout(RolloutCTX next_rollout) {

    State parent_s = state_registry.lookup_state(next_rollout.state_id);
    State curr_state = parent_s;
    vector<OperatorID> applicable_ops;
    successor_generator.generate_applicable_ops(curr_state, applicable_ops);
    vector<RolloutNode> curr_rollout_path;

    int rollout_limit = next_rollout.r_length;
    do {
        OperatorID op_id = *rng->choose(applicable_ops);
        OperatorProxy op = task_proxy.get_operators()[op_id];
        curr_state = state_registry.get_successor_state(curr_state, op);
        
        applicable_ops.clear();
        successor_generator.generate_applicable_ops(curr_state, applicable_ops);
        if (applicable_ops.size() == 0) {
            return make_pair(DEADEND, curr_rollout_path);
        }

        curr_rollout_path.push_back(RolloutNode(curr_state.get_id(), op_id));
        if (task_properties::is_goal_state(task_proxy, curr_state)) {
            return make_pair(GOAL, curr_rollout_path); // end of recursion: success
        }

        rollout_limit-=1;
    } while(rollout_limit > 0);

    SearchNode curr_node = search_space.get_node(curr_state);
    EvaluationContext succ_eval_context(
            curr_state, &statistics, false);
    int eval = succ_eval_context.get_evaluator_value_or_infinity(evaluator.get());
    if (eval == EvaluationResult::INFTY) {
        return make_pair(DEADEND, curr_rollout_path);
    }
    curr_rollout_path.back().h = eval;

    if (eval < next_rollout.h) { // if the rollout found an improvement
        return make_pair(greedy_rollout(curr_state, curr_rollout_path), curr_rollout_path);

    } else {
        return make_pair(UHR, curr_rollout_path);

    }

}


SearchStatus NewSearch::step() {
    
    RolloutCTX next_rollout = search_tree.select_next_state();
    statistics.inc_expanded(); // number of 'expanded' counts total rollouts in this search
    pair<RolloutResult, vector<RolloutNode>> result_pair = random_rollout(next_rollout);
    RolloutResult result = result_pair.first;
    if (result == GOAL) {
        open_states_along_rollout_path(next_rollout, result_pair.second);
        Plan plan;
        State state = state_registry.lookup_state(result_pair.second.back().state_id);
        search_space.trace_path(state, plan);
        set_plan(plan);
        return SOLVED;
    } else if (result == DEADEND) {
        // if deadend, try do another search step. 
        return IN_PROGRESS;
    } else {
        if (result == HI) {
            // add 'progressive' state to sparse search tree
            open_states_along_rollout_path(next_rollout, result_pair.second);
            StateID state_id_to_add = result_pair.second.back().state_id;
            int h = result_pair.second.back().h;
            search_tree.add_state(state_id_to_add, next_rollout.state_d + 1, h); // one deeper than parent for HI
            statistics.print_checkpoint_line(next_rollout.r_length);
            return IN_PROGRESS;
        } else { // UHR
            if (rng->random() < epsilon) {
                // add exploratory checkpoint to sparse search tree
                open_states_along_rollout_path(next_rollout, result_pair.second);
                StateID state_id_to_add = result_pair.second.back().state_id;
                int h = result_pair.second.back().h;
                search_tree.add_state(state_id_to_add, max(next_rollout.state_d-1, 0), h); // same depth as parent because no improvement
// cout << "HHHHHHH:::::::      " << h << endl;
                statistics.print_checkpoint_line(-1*next_rollout.r_length);
            }
            return IN_PROGRESS;
        }
    }
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
