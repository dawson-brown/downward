#ifndef SEARCH_ENGINES_NEW_SEARCH_H
#define SEARCH_ENGINES_NEW_SEARCH_H

#include "../open_list.h"
#include "../search_engine.h"
#include "../utils/rng.h"
#include "../task_utils/task_properties.h"

#include <memory>
#include <vector>
#include <map>
#include <math.h>

class Evaluator;
class PruningMethod;

namespace plugins {
class Feature;
}

#define TODO 1
#define GREED_EXPAND 2
#define RAND_EXPAND 4
#define DONE 8

namespace new_search {
class NewSearch : public SearchEngine {
    const bool reopen_closed_nodes;

    std::shared_ptr<Evaluator> evaluator;
    std::shared_ptr<utils::RandomNumberGenerator> rng;
    
    // struct KnownState {
    //     int depth;
    //     int op_index;
    //     int status;
    //     std::vector<OperatorID> operators;
    //     KnownState(int depth, int op_index, int status, std::vector<OperatorID> &operators)
    //         : depth(depth), op_index(op_index), status(status), operators(operators) {
    //     }
    //     KnownState() : // whats this?
    //         depth(0), op_index(-1), status(TODO) {} 
    // };
   
    // double current_sum;
    // double tau;
    // PerStateInformation<KnownState> known_states;
    // std::map<int, std::vector<std::vector<StateID>>, std::greater<int>> type_buckets;
    // int rollout_insert_depth;
    // int rollout_start_index;      

    void start_f_value_statistics(EvaluationContext &eval_context);
    void update_f_value_statistics(EvaluationContext &eval_context);
    void reward_progress();

protected:
    virtual void initialize() override;

    struct RolloutNode {
        StateID state_id;
        int g;

        RolloutNode(StateID state_id, int g):
            state_id(state_id), g(g) {}
    };
    struct RolloutCTX {
        StateID state_id;
        int r_length;
        int state_d;
        int h;

        RolloutCTX(StateID state_id, int r_length, int state_d, int h) :
            state_id(state_id), r_length(r_length), state_d(state_d), h(h) {}
    };
    struct SparseSearchTree {
        struct TreeNode {
            StateID state_id;
            int num_rollouts;
            int h;
            std::shared_ptr<utils::RandomNumberGenerator> rng;

            TreeNode(StateID state_id, int num_rollouts, int h, std::shared_ptr<utils::RandomNumberGenerator> rng) :
                state_id(state_id), num_rollouts(num_rollouts), h(h), rng(rng) {}
            TreeNode(StateID state_id, int h, std::shared_ptr<utils::RandomNumberGenerator> rng) :
                state_id(state_id), num_rollouts(0), h(h), rng(rng) {}

            inline int rl_lower_bound() {
                return floor(std::log(num_rollouts));
            }
            inline int rl_upper_bound() {
                return num_rollouts;
            }

            inline int rl_length() {
                return rng->random(
                    rl_lower_bound(),
                    rl_upper_bound()
                );
            }
        };

        double current_sum;
        double tau;
        std::shared_ptr<utils::RandomNumberGenerator> rng;
        std::map<int, std::vector<TreeNode>, std::greater<int>> depth_buckets;

        SparseSearchTree(double current_sum, double tau, std::shared_ptr<utils::RandomNumberGenerator> rng) 
            : current_sum(current_sum), tau(tau), rng(rng) {};

        RolloutCTX select_next_state() { 
            int selected_depth;
            int chosen_type_i;

            selected_depth = depth_buckets.begin()->first;
            if (depth_buckets.size() > 1) {
                double r = rng->random();
                
                double total_sum = current_sum;
                double p_sum = 0.0;
                for (auto it : depth_buckets) {
                    double p = 1.0 / total_sum;
                    p *= std::exp(-1.0*static_cast<double>(it.first) / tau); //remove -1.0 *
                    p *= static_cast<double>(it.second.size());
                    p_sum += p;
                    if (r <= p_sum) {
                        selected_depth = it.first;
                        break;
                    }
                }
            }

            vector<TreeNode> &states = depth_buckets.at(selected_depth);
            int chosen_i = rng->random(states.size());
            TreeNode node = states[chosen_i];

            node.num_rollouts+=1;
            return RolloutCTX(node.state_id, node.rl_length(), selected_depth, node.h);
        };
        void add_initial_state(StateID state_id, int h) {
            add_state(state_id, 0, h);
        }
        void add_state(StateID state_id, int depth, int h) {
            current_sum += std::exp(static_cast<double>(depth) / tau);
            depth_buckets[depth].push_back(
                TreeNode(state_id, h, rng)
            );
        }
    };
    SparseSearchTree search_tree;

    // type system manipulation
    // struct StateLoc {
    //     int depth;
    //     int breadth;
    //     int i;

    //     StateLoc(int depth, int breadth, int i) :
    //         depth(depth), breadth(breadth), i(i) {}
    //     StateLoc() : depth(0), breadth(0), i(0) {}
    // };
    // StateID select_next_state(StateLoc& loc);
    // void remove_from_type_buckets(StateLoc loc);
    // void insert_into_type_buckets(StateID s_id, StateLoc& loc);

    // rollout stuff
    enum RolloutResult { UHR, DEADEND, HI, GOAL }; 
    OperatorID random_next_action(State s);
    // OperatorID get_next_rollout_start(State s);
    bool add_state(const StateID s_id, const OperatorID op_id, const StateID parent_s_id);
    pair<RolloutResult, StateID> greedy_rollout(const State rollout_state);
    pair<RolloutResult, StateID> random_rollout();

    RolloutCTX curr_ctx;
    std::vector<RolloutNode> curr_rollout_path;

    // main search step
    virtual SearchStatus step() override;

public:
    explicit NewSearch(const plugins::Options &opts);
    virtual ~NewSearch() = default;

    virtual void print_statistics() const override;

    void dump_search_space() const;
};

extern void add_options_to_feature(plugins::Feature &feature);
}

#endif
