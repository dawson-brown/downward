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

SearchStatus SparseMCTS::step()
{

    shared_ptr<Node> selected = root;
    while (selected != select(selected)) {}
    simulate(selected);
    back_propogate(selected);

    return IN_PROGRESS;
}

void SparseMCTS::dump_search_space() const {
    search_space.dump(task_proxy);
}


void add_options_to_feature(plugins::Feature &feature) {
    SearchAlgorithm::add_options_to_feature(feature);
}
}
