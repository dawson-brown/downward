#ifndef SEARCH_ALGORITHMS_SPARSE_MCTS_H
#define SEARCH_ALGORITHMS_SPARSE_MCTS_H

#include "../search_algorithm.h"
#include "../utils/rng.h"

#include <memory>
#include <vector>
#include <random>
#include <map>

class Evaluator;

namespace plugins {
class Feature;
}

namespace sparse_lwm_search {
class SparseLWM : public SearchAlgorithm {

    std::shared_ptr<Evaluator> heuristic;
    std::shared_ptr<utils::RandomNumberGenerator> rng;

    double tau;
    double epsilon;
    double current_sum;

protected:

    struct NodeLoc {
        int depth;
        int i;

        NodeLoc(int depth, int i) : depth(depth), i(i) {};
        NodeLoc() : depth(-1), i(-1) {};
    };

    struct Node {
        StateID id;
        int h = std::numeric_limits<int>::max();
        int rollout_step = 0;
        int rl_length = 0;

        Node(StateID id, int h) : id(id), h(h) {}
        Node() : id(StateID::no_state) {}

        void set_rollout_length() {
            // int floor_log = static_cast <int> (std::floor(logb(rollout_step)));
            // int range = rollout_step - floor_log + 1;
            // int num = std::rand() % range + floor_log;
            // rl_length = std::max(num, 0);
            rl_length = std::rand() % (rollout_step+1) + 1;
        }
    };
    std::map<int, std::vector<Node>, std::greater<int>> depth_buckets;
    // utils::HashMap<int, std::pair<int, int>> partition_to_id_pair;

    PerStateInformation<Node> seen_states;
    bool is_dead_end(EvaluationContext &eval_context) const;
    virtual void initialize() override;

    NodeLoc cached_select = NodeLoc(-1, -1);
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

    NodeLoc select();
    Outcome expand(SparseLWM::Node &node, std::vector<OperatorID> &path);
    Outcome simulate(SparseLWM::Node &node, std::vector<OperatorID> &path);
    virtual SearchStatus step() override;
    Node open_path_to_new_node(NodeLoc selected, 
                                std::vector<OperatorID> &path, 
                                Outcome oc, bool bump);

public:
    explicit SparseLWM(const plugins::Options &opts);
    virtual ~SparseLWM() = default;

    virtual void print_statistics() const override;

    void dump_search_space() const;
};
}

#endif
