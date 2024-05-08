#ifndef SEARCH_ALGORITHMS_SPARSE_MCTS_H
#define SEARCH_ALGORITHMS_SPARSE_MCTS_H

#include "../search_algorithm.h"
#include "../utils/rng.h"

#include <memory>
#include <vector>
#include <random>

class Evaluator;

namespace plugins {
class Feature;
}

namespace sparse_mcts_search {
class SparseMCTS : public SearchAlgorithm {

    std::shared_ptr<Evaluator> heuristic;
    std::shared_ptr<utils::RandomNumberGenerator> rng;

    float c;
    float epsilon;
    float theta;

protected:

    struct Node {
        StateID id;
        OperatorID op_id;
        std::shared_ptr<Node> parent;
        std::vector<std::shared_ptr<Node>> children;
        int h = std::numeric_limits<int>::max();
        int num_visits = 0;
        float score = 0.0F;
        int rollout_step = 0;
        int rl_length = 0;

        Node(StateID id, OperatorID op_id, std::shared_ptr<Node> parent, int h)
            : id(id), op_id(op_id), parent(parent), h(h) {}

        void update(float score) {
            this->score += score;
            num_visits++;
        }

        void add_child(const std::shared_ptr<Node>& child) { 
            children.push_back(child); 
        }

        float get_avg_score() {
            return score / num_visits;
        }

        float utc(float c, int p_n) {
            if (num_visits == 0) {
                return std::numeric_limits<float>::max();
            }
            double l = std::log(p_n);
            float tmp = (float)sqrt( l / num_visits);
            return get_avg_score() + c * tmp;
        }

        void set_rollout_length() {
            int floor_log = static_cast <int> (std::floor(logb(rollout_step)));
            int range = rollout_step - floor_log + 1;
            int num = std::rand() % range + floor_log;
            rl_length = std::max(num, 0);
        }
    };
    std::shared_ptr<Node> root;

    bool is_dead_end(EvaluationContext &eval_context) const;
    virtual void initialize() override;

    std::shared_ptr<Node> cached_select = nullptr;
    enum Result { DEADEND = -1, UHR, HI, GOAL };
    struct Outcome {
        Result result;
        int h;

        Outcome(Result result, int h) :
            result(result), h(h) {}
        Outcome(Result result) :
            result(result), h(std::numeric_limits<int>::max()) {}
        Outcome() :
            result(UHR), h(std::numeric_limits<int>::max()) {}
    };
    std::shared_ptr<SparseMCTS::Node> select(std::shared_ptr<SparseMCTS::Node> node);
    Outcome simulate(SparseMCTS::Node &node, std::vector<OperatorID> &path);
    void back_propogate(Result result, SparseMCTS::Node &node);
    virtual SearchStatus step() override;

    std::shared_ptr<SparseMCTS::Node> open_path_to_new_node(std::shared_ptr<SparseMCTS::Node> selected, std::vector<OperatorID> path, Outcome oc);
    // void add_mcts_node(Node& selected, shared_ptr<Node> new_node);

public:
    explicit SparseMCTS(const plugins::Options &opts);
    virtual ~SparseMCTS() = default;

    virtual void print_statistics() const override;

    void dump_search_space() const;
};
}

#endif
