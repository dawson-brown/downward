#ifndef SEARCH_ENGINES_MC_SEARCH_H
#define SEARCH_ENGINES_MC_SEARCH_H

#include "../open_list.h"
#include "../search_engine.h"
#include "../utils/rng.h"

#include <memory>
#include <vector>
#include <queue>

class Evaluator;

namespace plugins {
class Feature;
}

enum RolloutScores {LOSE = -1, TIE, WIN, SOLVE};

namespace mc_search {
class MonteCarloSearch : public SearchEngine {

    // const bool reopen_closed_nodes;
    std::shared_ptr<Evaluator> evaluator;
    std::shared_ptr<utils::RandomNumberGenerator> rng;
    double explore_param;

    struct MC_RolloutAction {
        int edge_cost;
        OperatorID gen_op_id;
        StateID gen_state_id;
        MC_RolloutAction(int edge_cost, OperatorID gen_op_id, StateID gen_state_id) :
            edge_cost(edge_cost), gen_op_id(gen_op_id), gen_state_id(gen_state_id) {}
    };
    struct MCTS_State {
        bool is_fully_expanded;
        int next_child_i;
        int num_sims;
        int num_wins;
        std::vector<int> children_ids;
        MCTS_State(
            bool is_fully_expanded,
            int next_child_i,
            int num_sims,
            int num_wins
        ) : is_fully_expanded(is_fully_expanded), next_child_i(next_child_i), 
            num_sims(num_sims), num_wins(num_wins) {}
        MCTS_State() : is_fully_expanded(false), next_child_i(0), num_sims(0),
            num_wins(0), children_ids() {}
    };
    struct MCTS_node {
        int node_id;
        int parent_id;
        bool is_fully_expanded;
        int next_child_i;
        int num_sims;
        int num_wins;
        int eval;
        int g;
        StateID state;
        OperatorID gen_op_id;
        std::vector<int> children_ids;
        
        MCTS_node(
            int node_id,
            int parent_id,
            bool is_fully_expanded,
            int next_child_i,
            int num_sims,
            int num_wins,
            StateID state,
            OperatorID gen_op_id
        ) : node_id(node_id), parent_id(parent_id), is_fully_expanded(is_fully_expanded), next_child_i(next_child_i), 
            num_sims(num_sims), num_wins(num_wins), state(state), gen_op_id(gen_op_id) {}

        inline double uct(const double ex_param) const {
            if (num_sims == 0) return std::numeric_limits<double>::max();

            double winrate = ((double) num_wins) / ((double) num_sims);
            return winrate + ex_param * sqrt(std::log((double) num_sims) / ((double) num_sims));
        };

        inline int max_uct_child(const utils::HashMap<int, MCTS_node> &monte_carlo_tree, const double ex_param) const {
            int max_id = children_ids[0]; // yeah
            double max_uct = 0;
            for (int id: children_ids) { // im lazy
                double curr_uct = monte_carlo_tree.at(id).uct(ex_param);
                if (curr_uct > max_uct) {
                    max_id = id;
                    max_uct = curr_uct;
                }
            }
            return max_id;
        };
    };
    PerStateInformation<MCTS_State> mc_tree_data;
    PerStateInformation<int> mct_id;  // change int to pointer later
    utils::HashMap<int, MCTS_node> monte_carlo_tree; // and make this a pointer tree
    int root_id = 0;
    int next_id = 0;

protected:
    inline bool is_dead_end(EvaluationContext &eval_context);
    virtual void initialize() override;
    int selection();
    SearchStatus expansion(int selected_id);
    bool h_improves(EvaluationContext &first, EvaluationContext &second);
    RolloutScores rollout(int expand_id, int curr_id);
    void backpropogation(int back_prop_id, int score);
    virtual SearchStatus step() override;

public:
    explicit MonteCarloSearch(const plugins::Options &opts);
    virtual ~MonteCarloSearch() = default;

    virtual void print_statistics() const override;

    void dump_search_space() const;
};

extern void add_options_to_feature(plugins::Feature &feature);
}

#endif