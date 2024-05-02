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
      c(opts.get<float>("c")),
      epsilon(opts.get<float>("epsilon")) {}



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

    }

    print_initial_evaluator_values(eval_context);

}

void SparseMCTS::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
}

shared_ptr<SparseMCTS::Node> SparseMCTS::select(shared_ptr<SparseMCTS::Node> node)
{   
    shared_ptr<Node> best = node;
    if (cached_select != nullptr) {
        best = cached_select;
    }
    float bestScore = node->get_avg_score() + c * (float)sqrt( std::log(node->parent->num_visits) / node->num_visits);

    auto& children = node->children;
    // Use the UCT formula for selection
    for (auto& n : children) {
        float score = n->get_avg_score() + c * (float)sqrt( std::log(node->num_visits) / n->num_visits);

        if (score > bestScore) {
            bestScore = score;
            best = n;
        }
    }

    return best; // will return *node* if none of children are better
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
        // curr_state = parent_s.get_unregistered_successor(op);
        
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
    while (selected != select(selected)) {}

    vector<OperatorID> rollout_path;
    Outcome oc = simulate(*selected, rollout_path);

    if (oc.result == GOAL) {

        return SOLVED;
    }

    if (oc.result != DEADEND) {
         if (oc.result == HI) {
            cached_select = selected;
            open_path_to_new_node(selected, rollout_path, oc);
            // add_mcts_node(*selected, new_node);
         } else {
            cached_select = nullptr;
            if (rng->random() >= epsilon) {
                open_path_to_new_node(selected, rollout_path, oc);
                // add_mcts_node(*selected, new_node);
            }
        }
    }
    back_propogate(oc.result, *selected);
    return IN_PROGRESS;
}

void SparseMCTS::open_path_to_new_node(shared_ptr<Node> selected, std::vector<OperatorID> path, Outcome oc) {
    State state = state_registry.lookup_state(selected->id);
    OperatorID last_op(OperatorID::no_operator);
    for (OperatorID op_id : path) {
        OperatorProxy op = task_proxy.get_operators()[op_id];
        State succ_state = state_registry.get_successor_state(state, op);

        SearchNode node = search_space.get_node(state);
        SearchNode succ_node = search_space.get_node(succ_state);

        if (succ_node.is_new()) {
            succ_node.open(node, op, get_adjusted_cost(op));
        } else if (succ_node.get_g() > node.get_g() + get_adjusted_cost(op)) {
            succ_node.update_parent(node, op, get_adjusted_cost(op));
        }

        state = succ_state;
        last_op = op_id;
    }

    shared_ptr<Node> node(new Node(
        state.get_id(),
        last_op,
        selected,
        oc.h
    ));
    selected->add_child(node);
}

// void SparseMCTS::add_mcts_node(Node& selected, shared_ptr<Node> new_node) {

// }

void SparseMCTS::dump_search_space() const {
    search_space.dump(task_proxy);
}
}
