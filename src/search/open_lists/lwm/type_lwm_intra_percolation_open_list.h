#ifndef OPEN_LISTS_LWM_INTRA_PERCOLATION_OPEN_LIST_H
#define OPEN_LISTS_LWM_INTRA_PERCOLATION_OPEN_LIST_H

#include "../../open_list_factory.h"

#include "../../plugins/plugin.h"


/*
  Bench Transition System Entry State Open List. 
*/

namespace type_lwm_intra_percolation_open_list {
class LWMIntraPercolationOpenListFactory : public OpenListFactory {
    plugins::Options options;
public:
    explicit LWMIntraPercolationOpenListFactory(const plugins::Options &options);
    virtual ~LWMIntraPercolationOpenListFactory() override = default;

    virtual std::unique_ptr<StateOpenList> create_state_open_list() override;
    virtual std::unique_ptr<EdgeOpenList> create_edge_open_list() override;
};
}

#endif