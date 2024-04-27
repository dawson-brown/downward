#ifndef SEARCH_ALGORITHMS_SPARSE_MCTS_H
#define SEARCH_ALGORITHMS_SPARSE_MCTS_H

#include "../open_list.h"
#include "../search_algorithm.h"
#include "../utils/rng.h"

#include <memory>
#include <vector>

class Evaluator;

namespace plugins {
class Feature;
}

namespace sparse_mcts_search {
class SparseMCTS : public SearchAlgorithm {

    std::shared_ptr<Evaluator> heuristic;
    std::shared_ptr<utils::RandomNumberGenerator> rng;

    void reward_progress();

protected:
    bool is_dead_end(EvaluationContext &eval_context) const;
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit SparseMCTS(const plugins::Options &opts);
    virtual ~SparseMCTS() = default;

    virtual void print_statistics() const override;

    void dump_search_space() const;
};

extern void add_options_to_feature(plugins::Feature &feature);
}

#endif
