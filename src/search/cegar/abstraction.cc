#include "abstraction.h"

#include "abstract_state.h"
#include "refinement_hierarchy.h"
#include "transition_system.h"
#include "utils.h"

#include "../globals.h"

#include "../task_utils/task_properties.h"
#include "../utils/logging.h"
#include "../utils/math.h"
#include "../utils/memory.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>

using namespace std;

namespace cegar {
Abstraction::Abstraction(const shared_ptr<AbstractTask> &task, bool debug)
    : transition_system(utils::make_unique_ptr<TransitionSystem>(TaskProxy(*task).get_operators())),
      concrete_initial_state(TaskProxy(*task).get_initial_state()),
      goal_facts(task_properties::get_fact_pairs(TaskProxy(*task).get_goals())),
      init(nullptr),
      refinement_hierarchy(utils::make_unique_ptr<RefinementHierarchy>(task)),
      debug(debug) {
    create_trivial_abstraction(get_domain_sizes(TaskProxy(*task)));
}

Abstraction::~Abstraction() {
    for (AbstractState *state : states)
        delete state;
}

unique_ptr<RefinementHierarchy> Abstraction::extract_refinement_hierarchy() {
    assert(refinement_hierarchy);
    return move(refinement_hierarchy);
}

int Abstraction::get_init_h() const {
    // The initial state always holds its exact abstract goal distance.
    return init->get_goal_distance_estimate();
}

void Abstraction::mark_all_states_as_goals() {
    goals.clear();
    for (const AbstractState *state : states) {
        goals.insert(state->get_id());
    }
}

bool Abstraction::is_goal(AbstractState *state) const {
    return goals.count(state->get_id()) == 1;
}

void Abstraction::create_trivial_abstraction(const vector<int> &domain_sizes) {
    init = AbstractState::get_trivial_abstract_state(
        domain_sizes, refinement_hierarchy->get_root());
    transition_system->initialize(init);
    goals.insert(init->get_id());
    states.push_back(init);
}

void Abstraction::refine(AbstractState *state, int var, const vector<int> &wanted) {
    if (debug)
        cout << "Refine " << *state << " for " << var << "=" << wanted << endl;

    // Reuse state ID from obsolete parent to obtain consecutive IDs.
    int left_state_id = state->get_id();
    int right_state_id = get_num_states();
    pair<AbstractState *, AbstractState *> new_states = state->split(
        var, wanted, left_state_id, right_state_id);
    AbstractState *v1 = new_states.first;
    AbstractState *v2 = new_states.second;

    transition_system->rewire(states, state, v1, v2, var);

    states[v1->get_id()] = v1;
    assert(static_cast<int>(states.size()) == v2->get_id());
    states.push_back(v2);

    /*
      Due to the way we split the state into v1 and v2, v2 is never the new
      initial state and v1 is never a goal state.
    */
    if (state == init) {
        if (v1->includes(concrete_initial_state)) {
            assert(!v2->includes(concrete_initial_state));
            init = v1;
        } else {
            assert(v2->includes(concrete_initial_state));
            init = v2;
        }
        if (debug) {
            cout << "New init state #" << init->get_id() << ": " << *init << endl;
        }
    }

    if (is_goal(state)) {
        goals.erase(state->get_id());
        if (v1->includes(goal_facts)) {
            goals.insert(v1->get_id());
        }
        if (v2->includes(goal_facts)) {
            goals.insert(v2->get_id());
        }
        if (debug) {
            cout << "goal states: " << goals.size() << endl;
        }
    }

    delete state;
}

void Abstraction::print_statistics() const {
    cout << "States: " << get_num_states() << endl;
    cout << "Goal states: " << goals.size() << endl;
    transition_system->print_statistics();
}
}
