#include "sparse_mcts.h"

#include "../evaluation_context.h"
#include "../evaluator.h"
#include "../open_list_factory.h"
#include "../pruning_method.h"

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
      c(opts.get<double>("c")),
      epsilon(opts.get<double>("epsilon"))  {}

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

        // TODO: set root

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

SparseMCTS::Outcome SparseMCTS::simulate(SparseMCTS::Node &node) {
    StateID start_id = node.id;
    Outcome oc;

    StateRegistry rollout_registry(task_proxy);
    // do rollout like other branch...


}


void SparseMCTS::back_propogate(Result result, SparseMCTS::Node &node) {

    float score = 1 ? result == HI : 0;
    node.update(score);

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
    Outcome oc = simulate(*selected);

    if (oc.result == GOAL) {

        return SOLVED;
    }

    if (oc.result != DEADEND) {
         if (oc.result == HI) {
            cached_select = selected;
                // TODO: add oc.terminal to tree
                // -- this involves opening path in state registry
                // and including the Node into the child list and all that.
         } else {
            cached_select = nullptr;
            if (rng->random() >= epsilon) {
                // TODO: add oc.terminal to tree
                // -- this involves opening path in state registry
                // and including the Node into the child list and all that.
            }
        }
    }
    back_propogate(oc.result, *selected);
    return IN_PROGRESS;
}

void SparseMCTS::dump_search_space() const {
    search_space.dump(task_proxy);
}


void add_options_to_feature(plugins::Feature &feature) {
    SearchAlgorithm::add_options_to_feature(feature);
}
}
