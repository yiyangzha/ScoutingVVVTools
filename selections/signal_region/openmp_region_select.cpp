#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <mutex>
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

bool picks_less(const int* picks,
                const std::array<int, kMaxRegions>& best,
                int count) {
    for (int i = 0; i < count; ++i) {
        if (picks[i] != best[i]) {
            return picks[i] < best[i];
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

void configure_threads(int num_threads) {
#ifdef _OPENMP
    const int available_threads = std::max(1, omp_get_num_procs());
    const int thread_limit = std::max(1, std::min(num_threads, available_threads));
    omp_set_num_threads(thread_limit);
#else
    (void)num_threads;
#endif
}

State run_beam_selection(
    int n_candidates,
    int dim,
    int target_regions,
    int beam_width,
    const double* lows,
    const double* highs,
    const double* z2
) {
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
        State none;
        return none;
    }
    return best;
}

bool compatible_picks(
    const int* picks,
    int count,
    const double* lows,
    const double* highs,
    int dim,
    int candidate
) {
    for (int i = 0; i < count; ++i) {
        if (overlaps(lows, highs, dim, candidate, picks[i])) {
            return false;
        }
    }
    return true;
}

struct SearchContext {
    int n_candidates = 0;
    int dim = 0;
    int target = 0;
    const double* lows = nullptr;
    const double* highs = nullptr;
    const double* z2 = nullptr;
    long long max_nodes = 0;
    double time_limit_seconds = 0.0;
    std::chrono::steady_clock::time_point start_time;

    std::atomic<long long> nodes{0};
    std::atomic<bool> stop{false};
    std::atomic<double> best_score{0.0};

    std::mutex best_mutex;
    std::array<int, kMaxRegions> best_picks{};
    bool found = false;
};

bool time_limit_reached(const SearchContext& ctx) {
    if (ctx.time_limit_seconds <= 0.0) {
        return false;
    }
    const auto now = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(now - ctx.start_time).count();
    return elapsed >= ctx.time_limit_seconds;
}

bool should_stop(SearchContext& ctx, long long node_count) {
    if (ctx.stop.load(std::memory_order_relaxed)) {
        return true;
    }
    if (ctx.max_nodes > 0 && node_count >= ctx.max_nodes) {
        ctx.stop.store(true, std::memory_order_relaxed);
        return true;
    }
    if (time_limit_reached(ctx)) {
        ctx.stop.store(true, std::memory_order_relaxed);
        return true;
    }
    return false;
}

double optimistic_bound(
    const SearchContext& ctx,
    int start,
    const int* picks,
    int depth,
    double score
) {
    const int remaining = ctx.target - depth;
    if (remaining <= 0) {
        return score;
    }
    double bound = score;
    int count = 0;
    for (int cand = start; cand < ctx.n_candidates; ++cand) {
        if (compatible_picks(picks, depth, ctx.lows, ctx.highs, ctx.dim, cand)) {
            bound += ctx.z2[cand];
            ++count;
            if (count >= remaining) {
                return bound;
            }
        }
    }
    return -std::numeric_limits<double>::infinity();
}

void update_best(SearchContext& ctx, const int* picks, double score) {
    constexpr double eps = 1.0e-12;
    const double current = ctx.best_score.load(std::memory_order_relaxed);
    if (score < current - eps) {
        return;
    }
    std::lock_guard<std::mutex> lock(ctx.best_mutex);
    const double best_now = ctx.best_score.load(std::memory_order_relaxed);
    if (!ctx.found ||
        score > best_now + eps ||
        (std::abs(score - best_now) <= eps && picks_less(picks, ctx.best_picks, ctx.target))) {
        ctx.found = true;
        ctx.best_score.store(score, std::memory_order_relaxed);
        for (int i = 0; i < ctx.target; ++i) {
            ctx.best_picks[i] = picks[i];
        }
    }
}

void dfs_branch_bound(
    SearchContext& ctx,
    int start,
    int depth,
    double score,
    int* picks
) {
    if (ctx.stop.load(std::memory_order_relaxed)) {
        return;
    }

    const long long node_count = ctx.nodes.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((node_count & 4095LL) == 0LL && should_stop(ctx, node_count)) {
        return;
    }

    if (depth == ctx.target) {
        update_best(ctx, picks, score);
        return;
    }

    const double bound = optimistic_bound(ctx, start, picks, depth, score);
    const double best = ctx.best_score.load(std::memory_order_relaxed);
    if (!std::isfinite(bound) || bound <= best + 1.0e-12) {
        return;
    }

    const int remaining = ctx.target - depth;
    const int last = ctx.n_candidates - remaining;
    for (int cand = start; cand <= last; ++cand) {
        if (ctx.stop.load(std::memory_order_relaxed)) {
            return;
        }
        if (!compatible_picks(picks, depth, ctx.lows, ctx.highs, ctx.dim, cand)) {
            continue;
        }
        picks[depth] = cand;
        dfs_branch_bound(ctx, cand + 1, depth + 1, score + ctx.z2[cand], picks);
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

    configure_threads(num_threads);
    const State best = run_beam_selection(
        n_candidates, dim, target_regions, beam_width, lows, highs, z2
    );
    if (best.count == 0) {
        return 0;
    }

    for (int i = 0; i < target_regions; ++i) {
        out_indices[i] = (i < best.count) ? best.picks[i] : -1;
    }
    return best.count;
}

extern "C" int select_regions_branch_bound_openmp(
    int n_candidates,
    int dim,
    int target_regions,
    int beam_width,
    long long max_nodes,
    double time_limit_seconds,
    const double* lows,
    const double* highs,
    const double* z2,
    int* out_indices,
    double* out_stats,
    int num_threads
) {
    if (n_candidates <= 0 || dim <= 0 || target_regions <= 0 || beam_width <= 0) {
        return -1;
    }
    if (target_regions > kMaxRegions) {
        return -2;
    }
    if (n_candidates < target_regions) {
        return 0;
    }

    configure_threads(num_threads);

    SearchContext ctx;
    ctx.n_candidates = n_candidates;
    ctx.dim = dim;
    ctx.target = target_regions;
    ctx.lows = lows;
    ctx.highs = highs;
    ctx.z2 = z2;
    ctx.max_nodes = std::max(0LL, max_nodes);
    ctx.time_limit_seconds = std::max(0.0, time_limit_seconds);
    ctx.start_time = std::chrono::steady_clock::now();

    const State seed = run_beam_selection(
        n_candidates, dim, target_regions, beam_width, lows, highs, z2
    );
    if (seed.count == target_regions) {
        std::lock_guard<std::mutex> lock(ctx.best_mutex);
        ctx.found = true;
        ctx.best_score.store(seed.score, std::memory_order_relaxed);
        for (int i = 0; i < target_regions; ++i) {
            ctx.best_picks[i] = seed.picks[i];
        }
    }

    const int root_count = n_candidates - target_regions + 1;
    std::vector<double> root_bounds(root_count, -std::numeric_limits<double>::infinity());
    std::vector<unsigned char> root_done(root_count, 0);

    for (int root = 0; root < root_count; ++root) {
        int picks[kMaxRegions] = {};
        picks[0] = root;
        root_bounds[root] = optimistic_bound(ctx, root + 1, picks, 1, z2[root]);
    }

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int root = 0; root < root_count; ++root) {
        if (ctx.stop.load(std::memory_order_relaxed)) {
            continue;
        }
        const double bound = root_bounds[root];
        if (!std::isfinite(bound) ||
            bound <= ctx.best_score.load(std::memory_order_relaxed) + 1.0e-12) {
            root_done[root] = 1;
            continue;
        }
        int picks[kMaxRegions] = {};
        picks[0] = root;
        dfs_branch_bound(ctx, root + 1, 1, z2[root], picks);
        if (!ctx.stop.load(std::memory_order_relaxed)) {
            root_done[root] = 1;
        }
    }

    const bool completed = !ctx.stop.load(std::memory_order_relaxed);
    double best_score = ctx.best_score.load(std::memory_order_relaxed);
    double upper_bound = best_score;
    int roots_done = 0;
    if (completed) {
        upper_bound = best_score;
        roots_done = root_count;
    } else {
        for (int root = 0; root < root_count; ++root) {
            if (root_done[root]) {
                ++roots_done;
                continue;
            }
            if (std::isfinite(root_bounds[root])) {
                upper_bound = std::max(upper_bound, root_bounds[root]);
            }
        }
    }

    if (out_stats != nullptr) {
        out_stats[0] = best_score;
        out_stats[1] = upper_bound;
        out_stats[2] = static_cast<double>(ctx.nodes.load(std::memory_order_relaxed));
        out_stats[3] = completed ? 1.0 : 0.0;
        out_stats[4] = static_cast<double>(roots_done);
        out_stats[5] = static_cast<double>(root_count);
    }

    if (!ctx.found) {
        return 0;
    }

    for (int i = 0; i < target_regions; ++i) {
        out_indices[i] = ctx.best_picks[i];
    }
    return target_regions;
}
