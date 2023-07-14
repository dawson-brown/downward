#ifndef OPEN_LISTS_SOFTMIN_OPEN_LIST_H
#define OPEN_LISTS_SOFTMIN_OPEN_LIST_H

#include "../open_list_factory.h"
#include "../plugins/plugin.h"


namespace softmin_open_list {
class SoftminOpenListFactory : public OpenListFactory {
  plugins::Options options;

 public:
  explicit SoftminOpenListFactory(const plugins::Options &options);
  virtual ~SoftminOpenListFactory() override = default;

  virtual std::unique_ptr<StateOpenList> create_state_open_list() override;
  virtual std::unique_ptr<EdgeOpenList> create_edge_open_list() override;
};
}  

#endif
