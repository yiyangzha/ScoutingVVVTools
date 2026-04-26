#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kMaxRegions = 16;

struct State {
    std::array<int, kMaxRegions> picks{};
    int count = 0;
    double score = 0.0;
};

bool overlaps(
    const double* lows,
    const double* highs,
    int dim,
    int lhs,
    int rhs
) {
    constexpr double eps = 1.0e-12;
    const double* lo1 = lows + static_cast<long long>(lhs) * dim;
    const double* hi1 = highs + static_cast<long long>(lhs) * dim;
    const double* lo2 = lows + static_cast<long long>(rhs) * dim;
    const double* hi2 = highs + static_cast<long long>(rhs) * dim;
    for (int d = 0; d < dim; ++d) {
        if (!(lo1[d] < hi2[d] - eps && lo2[d] < hi1[d] - eps)) {
            return false;
        }
    }
    return true;
}

bool compatible(
    const State& state,
    const double* lows,
    const double* highs,
    int dim,
    int candidate
) {
    for (int i = 0; i < state.count; ++i) {
        if (overlaps(lows, highs, dim, candidate, state.picks[i])) {
            return false;
        }
    }
    return true;
}

bool better_state(const State& a, const State& b) {
    if (a.score != b.score) {
        return a.score > b.score;
    }
    if (a.count != b.count) {
        return a.count > b.count;
    }
    for (int i = 0; i < a.count && i < b.count; ++i) {
        if (a.picks[i] != b.picks[i]) {
            return a.picks[i] < b.picks[i];
        }
    }
    return false;
}

void prune(std::vector<State>& states, int target, int beam_width) {
    std::vector<std::vector<State>> buckets(target + 1);
    for (const auto& state : states) {
        if (state.count <= target) {
            buckets[state.count].push_back(state);
        }
    }

    states.clear();
    states.reserve(static_cast<size_t>(beam_width) * (target + 1));
    for (auto& bucket : buckets) {
        std::sort(bucket.begin(), bucket.end(), better_state);
        if (static_cast<int>(bucket.size()) > beam_width) {
            bucket.resize(beam_width);
        }
        states.insert(states.end(), bucket.begin(), bucket.end());
    }
}

}  // namespace

extern "C" int select_regions_beam_openmp(
    int n_candidates,
    int dim,
    int target_regions,
    int beam_width,
    const double* lows,
    const double* highs,
    const double* z2,
    int* out_indices,
    int num_threads
) {
    if (n_candidates <= 0 || dim <= 0 || target_regions <= 0 || beam_width <= 0) {
        return -1;
    }
    if (target_regions > kMaxRegions) {
        return -2;
    }

#ifdef _OPENMP
    const int available_threads = std::max(1, omp_get_num_procs());
    const int thread_limit = std::max(1, std::min(num_threads, available_threads));
    omp_set_num_threads(thread_limit);
#else
    (void)num_threads;
#endif

    std::vector<State> states;
    State empty;
    states.push_back(empty);

    for (int cand = 0; cand < n_candidates; ++cand) {
        const int n_states = static_cast<int>(states.size());

#ifdef _OPENMP
        const int n_threads = std::max(1, omp_get_max_threads());
        std::vector<std::vector<State>> local_new(n_threads);
#pragma omp parallel
        {
            const int tid = omp_get_thread_num();
            auto& mine = local_new[tid];
#pragma omp for schedule(static)
            for (int si = 0; si < n_states; ++si) {
                const State& state = states[si];
                if (state.count >= target_regions) {
                    continue;
                }
                if (!compatible(state, lows, highs, dim, cand)) {
                    continue;
                }
                State next = state;
                next.picks[next.count] = cand;
                next.count += 1;
                next.score += z2[cand];
                mine.push_back(next);
            }
        }
        for (auto& mine : local_new) {
            states.insert(states.end(), mine.begin(), mine.end());
        }
#else
        std::vector<State> local_new;
        for (int si = 0; si < n_states; ++si) {
            const State& state = states[si];
            if (state.count >= target_regions) {
                continue;
            }
            if (!compatible(state, lows, highs, dim, cand)) {
                continue;
            }
            State next = state;
            next.picks[next.count] = cand;
            next.count += 1;
            next.score += z2[cand];
            local_new.push_back(next);
        }
        states.insert(states.end(), local_new.begin(), local_new.end());
#endif

        prune(states, target_regions, beam_width);
    }

    State best;
    bool found = false;
    for (const auto& state : states) {
        if (!found ||
            state.count > best.count ||
            (state.count == best.count && better_state(state, best))) {
            best = state;
            found = true;
        }
    }

    if (!found || best.count == 0) {
        return 0;
    }

    for (int i = 0; i < target_regions; ++i) {
        out_indices[i] = (i < best.count) ? best.picks[i] : -1;
    }
    return best.count;
}
