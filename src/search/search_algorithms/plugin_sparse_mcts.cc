#include "sparse_mcts.h"
#include "search_common.h"

#include "../plugins/plugin.h"
#include "../utils/rng_options.h"

using namespace std;

namespace plugin_sparse_mcts_search {
class SparseMCTSFeature : public plugins::TypedFeature<SearchAlgorithm, sparse_mcts_search::SparseMCTS> {
public:
    SparseMCTSFeature() : TypedFeature("sparse_mcts") {
        document_title("Sparse Monte Carlo Tree Search");

        add_option<shared_ptr<Evaluator>>("eval", "evaluator");
        add_option<double>(
            "c",
            "MCTS exploration parameter",
            "1.414");
        add_option<double>(
            "epsilon",
            "probability of adding the terminal state of a failed rollout",
            "0.1",
        plugins::Bounds("0.0", "1.0"));
        // add_list_option<shared_ptr<Evaluator>>(
        //     "preferred",
        //     "use preferred operators of these evaluators",
        //     "[]");
        SearchAlgorithm::add_options_to_feature(*this);
        utils::add_rng_options(*this); 

    }
};

static plugins::FeaturePlugin<SparseMCTSFeature> _plugin;
}
