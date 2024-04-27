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

    double c;
    double epsilon;

protected:

    struct Node {
        StateID id;
        OperatorID op_id;
        std::shared_ptr<Node> parent;
        std::vector<std::shared_ptr<Node>> children;
        int num_visits = 0;
        float score = 0.0F;

        Node(StateID id, OperatorID op_id, std::shared_ptr<Node> parent)
            : id(id), op_id(op_id), parent(parent) {}

        void update(float score) {
            this->score += score;
            num_visits++;
        }

        void add_child(const std::shared_ptr<Node>& child) { 
            children.push_back(child); 
        }

        double get_avg_score() {
            return score / num_visits;
        }
    };
    shared_ptr<Node> root;

    bool is_dead_end(EvaluationContext &eval_context) const;
    virtual void initialize() override;

    shared_ptr<SparseMCTS::Node> select(shared_ptr<SparseMCTS::Node> node);
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
