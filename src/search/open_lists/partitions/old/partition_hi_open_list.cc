#include "partition_hi_open_list.h"

#include "partitions/partition_open_list.h"
#include "partitions/inter/partition_policy.h"
#include "partitions/intra/node_policy.h"
#include "../evaluator.h"
#include "../open_list.h"

#include "../plugins/plugin.h"
#include "../utils/collections.h"
#include "../utils/hash.h"
#include "../utils/markup.h"
#include "../utils/memory.h"
#include "../utils/rng.h"
#include "../utils/rng_options.h"

#include <memory>
#include <unordered_map>
#include <vector>

using namespace std;
using namespace partition_open_list;

namespace partition_hi_open_list {
template<class Entry>
class PartitionHIOpenList : public PartitionOpenList<Entry> {

    StateID cached_next_state_id = StateID::no_state;
    StateID cached_parent_id = StateID::no_state;
    int last_removed = StateID::no_state.get_value();

    int parent_h;
    int parent_partition_key = -1;

protected:
    virtual void do_insertion(
        EvaluationContext &eval_context, const Entry &entry) override;

public:
    Entry remove_min() override;
    explicit PartitionHIOpenList(const plugins::Options &opts);
    virtual ~PartitionHIOpenList() override = default;

    virtual void notify_initial_state(const State &initial_state) override;
    virtual void notify_state_transition(const State &parent_state,
                                         OperatorID op_id,
                                         const State &state) override;


};

template<class Entry>
void PartitionHIOpenList<Entry>::notify_initial_state(const State &initial_state) {
    cached_next_state_id = initial_state.get_id();
    parent_h = numeric_limits<int>::max();
}

template<class Entry>
void PartitionHIOpenList<Entry>::notify_state_transition(const State &parent_state,
                                        OperatorID op_id,
                                        const State &state)
{
    cached_next_state_id = state.get_id();
    cached_parent_id = parent_state.get_id();

    parent_partition_key = this->partitioned_nodes.at(this->cached_parent_id.get_value()).first.partition;
    parent_h = this->partitioned_nodes.at(cached_parent_id.get_value()).first.eval;
}

template<class Entry>
void PartitionHIOpenList<Entry>::do_insertion(
    EvaluationContext &eval_context, const Entry &entry) {
    
    int new_h = eval_context.get_evaluator_value_or_infinity(this->evaluator.get());
    int partition_key;
    bool new_type;
    if ( (new_h < parent_h) ) {
        partition_key = this->cached_next_state_id.get_value();
        new_type = true;
    } else {
        partition_key = parent_partition_key;
        new_type = false;
    }
    this->partition_selector->notify_partition_transition(
        parent_partition_key, 
        cached_parent_id.get_value(), 
        partition_key, 
        this->cached_next_state_id.get_value());

    PartitionOpenList<Entry>::partition_insert(this->cached_next_state_id.get_value(), new_h, entry, partition_key, new_type);

}

template<class Entry>
Entry PartitionHIOpenList<Entry>::remove_min() {

    if (last_removed != StateID::no_state.get_value()) {
        this->partition_selector->notify_removal(this->partitioned_nodes.at(last_removed).first.partition, last_removed);
        this->partitioned_nodes.erase(last_removed);
    }

    int chosen_partition = this->partition_selector->get_next_partition();
    int chosen_node = this->node_selector->get_next_node(chosen_partition);

    Entry result = this->partitioned_nodes.at(chosen_node).second;
    last_removed = chosen_node;
    return result;

}

template<class Entry>
PartitionHIOpenList<Entry>::PartitionHIOpenList(const plugins::Options &opts)
    : PartitionOpenList<Entry>(
        opts.get<shared_ptr<Evaluator>>("eval"),
        opts.get<shared_ptr<PartitionPolicy>>("partition_policy"),
        opts.get<shared_ptr<NodePolicy>>("node_policy")
    ) {}


PartitionHIOpenListFactory::PartitionHIOpenListFactory(
    const plugins::Options &options)
    : options(options) {
}

unique_ptr<StateOpenList>
PartitionHIOpenListFactory::create_state_open_list() {
    return utils::make_unique_ptr<PartitionHIOpenList<StateOpenListEntry>>(options);
}

unique_ptr<EdgeOpenList>
PartitionHIOpenListFactory::create_edge_open_list() {
    return utils::make_unique_ptr<PartitionHIOpenList<EdgeOpenListEntry>>(options);
}

class PartitionHIOpenListFeature : public plugins::TypedFeature<OpenListFactory, PartitionHIOpenListFactory> {
public:
    PartitionHIOpenListFeature() : TypedFeature("hi_partition") {
        document_title("Partition Low Water-mark Open List");
        document_synopsis("A configurable open list that selects nodes by first"
         "choosing a node parition and then choosing a node from within it."
         "The policies for insertion (choosing a partition to insert into or"
         "creating a new one) and selection (partition first, then a node within it)"
         "are  specified policies");
        add_option<shared_ptr<Evaluator>>("eval", "evaluator");
        add_option<shared_ptr<PartitionPolicy>>("partition_policy", "partition selection policy");
        add_option<shared_ptr<NodePolicy>>("node_policy", "node selection policy");
    }
};

static plugins::FeaturePlugin<PartitionHIOpenListFeature> _plugin;
}