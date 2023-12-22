#ifndef NODE_POLICIES_INTRA_RANDOM_MIN_H
#define NODE_POLICIES_INTRA_RANDOM_MIN_H

#include "node_policy.h"

#include "../../../evaluator.h"
#include "../../../utils/rng.h"
#include "../../../utils/rng_options.h"


namespace intra_partition_random_min {
class IntraRandomMinPolicy : public NodePolicy {

    std::shared_ptr<utils::RandomNumberGenerator> rng;
    
    struct HeapNode {
        int id;
        int h;
        HeapNode(int id, int h)
            : id(id), h(h) {
        }
        bool operator>(const HeapNode &other) const {
            return std::make_pair(h, id) > std::make_pair(other.h, other.id);
        }
    };
    utils::HashMap<int, std::vector<HeapNode>> partition_heaps;

public:
    explicit IntraRandomMinPolicy(const plugins::Options &opts);
    virtual ~IntraRandomMinPolicy() override = default;

    virtual void notify_insert(
        int partition_key,
        int node_key,
        bool new_partition,
        int eval
    ) override;
    virtual int get_next_node(int partition_key) override;
    virtual void clear() {
        partition_heaps.clear();
    };
};

}

#endif