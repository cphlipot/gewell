#include "gewell/pending_prefix.h"

#include <algorithm>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <utility>


#include <iostream>

namespace gewell::pending_prefix {
namespace {

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(std::string("pending prefix: ") + message);
}


bool run_self_tests(std::string* failure) {
  try {
    const auto make_prompt = [](std::uint32_t common,
                                std::initializer_list<std::uint32_t> suffix) -> Prompt {
      auto prompt = std::make_shared<std::vector<std::uint32_t>>(common, 7);
      prompt->insert(prompt->end(), suffix.begin(), suffix.end());
      return prompt;
    };
    const auto rejects = [](const auto& operation) {
      try { operation(); }
      catch (const std::exception&) { return true; }
      return false;
    };
    const auto advance = [](Work& work, std::uint32_t rows) {
      const auto step = work.begin_step(rows);
      require(step.begin == work.completed() && step.end > step.begin &&
                  step.end <= work.next_boundary() && work.inflight() && !work.at_boundary(),
              "dispatch did not preserve completed cursor and next boundary");
      work.complete_step(true);
      require(work.completed() == step.end && !work.inflight(), "completion lost dispatched progress");
      return step;
    };

    // Register waiters before restoring a retained prefix. Restore commits
    // existing state without inventing a GPU step or skipping a known branch.
    Work restored(make_prompt(4096, {10}), 1);
    const auto restored_waiter = make_prompt(3071, {20});
    require(restored.join(2, restored_waiter).joined,
            "retained-prefix waiter setup failed");
    require(rejects([&] { restored.restore_prefix(3072); }) &&
                restored.completed() == 0 && restored.dependents().size() == 2,
            "checkpoint restore crossed a pending branch");
    restored.restore_prefix(2048);
    require(restored.completed() == 2048 && !restored.inflight() &&
                !restored.failed() && !restored.at_boundary() &&
                restored.dependents().size() == 2 &&
                restored.dependents()[1].request_id == 2 &&
                restored.dependents()[1].prompt == restored_waiter &&
                restored.next_boundary() == 3071 &&
                rejects([&] { restored.complete_step(true); }) &&
                rejects([&] { restored.restore_prefix(3071); }),
            "checkpoint restore changed dependencies or invented an in-flight step");
    const auto restored_step = advance(restored, 2048);
    require(restored_step.begin == 2048 && restored_step.end == 3071 &&
                restored.at_boundary(), "restored work replayed its cached prefix");
    Work exact_restore(make_prompt(4096, {10}), 1);
    require(exact_restore.join(2, restored_waiter).joined,
            "exact-boundary restore setup failed");
    exact_restore.restore_prefix(3071);
    require(exact_restore.at_boundary() &&
                rejects([&] { (void)exact_restore.begin_step(1); }),
            "exact-boundary restore failed to expose ready dependents");
    Work busy_restore(make_prompt(4096, {}), 1);
    (void)busy_restore.begin_step(1024);
    require(rejects([&] { busy_restore.restore_prefix(2048); }) &&
                busy_restore.completed() == 0 && busy_restore.dispatched_end() == 1024,
            "restore replaced an in-flight chunk");
    busy_restore.complete_step(false);
    require(rejects([&] { busy_restore.restore_prefix(2048); }) &&
                busy_restore.failed() && busy_restore.completed() == 0,
            "restore revived failed work");

    // Cold bursts compute the common trunk once, including a nonaligned end.
    for (const std::uint32_t common : {4096U, 4095U}) {
      Work burst(make_prompt(common, {10, 11}), 1);
      const auto a = burst.join(2, make_prompt(common, {20}));
      const auto b = burst.join(3, make_prompt(common, {30}));
      require(a.joined && b.joined && a.common_tokens == common &&
                  b.common_tokens == common && burst.next_boundary() == common,
              "cold burst failed to share its exact trunk");
      std::uint32_t processed = 0, steps = 0;
      while (!burst.at_boundary()) {
        const auto step = advance(burst, 1024);
        processed += step.end - step.begin;
        ++steps;
      }
      require(processed == common && steps == 4 && burst.completed() == common,
              "shared trunk repeated work or rounded its boundary");
      require(rejects([&] { (void)burst.begin_step(1024); }),
              "dispatch crossed a ready fork boundary");
      require(burst.detach(2) && burst.detach(3) && !burst.at_boundary(),
              "detaching branches did not expose the remaining path");
      const auto suffix = advance(burst, 1024);
      require(suffix.begin == common && suffix.end == common + 2 && burst.at_boundary(),
              "remaining producer restarted its common prefix");
      require(burst.detach(1) && burst.releasable(), "finished burst remained owned");
    }

    Work staggered(make_prompt(4096, {10}), 1);
    const auto dispatched = staggered.begin_step(2048);
    require(dispatched.begin == 0 && dispatched.end == 2048 && staggered.completed() == 0,
            "dispatch prematurely committed tokens");
    const auto late = staggered.join(2, make_prompt(2047, {20}));
    require(!late.joined && late.common_tokens == 2047 && staggered.dependents().size() == 1,
            "late join changed an already dispatched boundary");
    require(staggered.join(3, make_prompt(2048, {30})).joined,
            "join exactly at dispatched end was rejected");
    require(rejects([&] { (void)staggered.begin_step(1); }), "double dispatch was accepted");
    staggered.complete_step(true);
    require(staggered.at_boundary() && staggered.completed() == 2048,
            "dispatched-end join lost its ready boundary");
    require(staggered.detach(3) && staggered.join(4, make_prompt(2048, {40})).joined &&
                staggered.at_boundary(), "idle join at completed cursor was rejected");
    require(!staggered.join(5, make_prompt(2047, {50})).joined,
            "idle join rewound completed progress");
    require(staggered.detach(4) && staggered.join(6, make_prompt(3071, {60})).joined,
            "staggered join ahead of completed cursor was rejected");
    require(advance(staggered, 2048).end == 3071,
            "staggered join did not shorten the next undispatched chunk");

    // Cancellation transfers the representative without changing work state.
    auto original = make_prompt(4096, {10});
    std::weak_ptr<const std::vector<std::uint32_t>> original_storage = original;
    Work handoff(original, 1);
    require(handoff.join(2, make_prompt(4096, {20})).joined &&
                handoff.join(3, make_prompt(4096, {30})).joined, "handoff setup failed");
    original.reset();
    advance(handoff, 1024);
    (void)handoff.begin_step(1024);
    require(handoff.cancel(1) && handoff.dependents().front().request_id == 2 &&
                handoff.completed() == 1024 && !original_storage.expired(),
            "producer cancellation lost owned prompt or reset progress");
    require(handoff.cancel(2) && handoff.dependents().front().request_id == 3,
            "successive cancellation lost the last waiter");
    handoff.complete_step(true);
    require(handoff.completed() == 2048 && !handoff.releasable(), "handoff restarted work");
    (void)handoff.begin_step(1024);
    require(handoff.cancel(3) && !handoff.releasable() && !original_storage.expired(),
            "last cancellation released work before synchronization");
    handoff.complete_step(true);
    require(handoff.releasable() && handoff.completed() == 3072,
            "last canceled step did not release after synchronization");
    Work canceled(make_prompt(10, {}), 1);
    require(canceled.cancel(1) && !canceled.cancel(1) && canceled.releasable() &&
                canceled.completed() == 0 && !canceled.at_boundary() &&
                rejects([&] { (void)canceled.begin_step(1); }),
            "idle final cancellation retained runnable work");
    Work idle_handoff(make_prompt(4096, {10}), 1);
    require(idle_handoff.join(2, make_prompt(4096, {20})).joined &&
                idle_handoff.join(3, make_prompt(4096, {30})).joined &&
                idle_handoff.cancel(1), "idle producer cancellation setup failed");
    advance(idle_handoff, 1024);
    require(idle_handoff.cancel(2) && idle_handoff.completed() == 1024 &&
                idle_handoff.dependents().front().request_id == 3,
            "idle successor cancellation restarted shared work");
    require(idle_handoff.cancel(3) && idle_handoff.releasable() &&
                idle_handoff.completed() == 1024,
            "idle final cancellation changed completed progress");

    std::weak_ptr<const std::vector<std::uint32_t>> canceled_storage;
    {
      Work last(make_prompt(32, {}), 1);
      canceled_storage = last.prompt();
      (void)last.begin_step(16);
      require(last.cancel(1) && !last.releasable() && !canceled_storage.expired(),
              "final canceled work lost its in-flight input storage");
      last.complete_step(false);
      require(last.failed() && last.completed() == 0 && last.releasable(),
              "failure after final cancellation committed tokens or retained work");
    }
    require(canceled_storage.expired(), "destroyed work retained canceled prompt storage");

    Work failed(make_prompt(4096, {}), 1);
    advance(failed, 1024);
    (void)failed.begin_step(1024);
    failed.complete_step(false);
    require(failed.failed() && !failed.inflight() && failed.completed() == 1024 &&
                !failed.at_boundary() && !failed.join(2, make_prompt(4096, {})).joined &&
                rejects([&] { (void)failed.begin_step(1); }) &&
                rejects([&] { failed.complete_step(true); }),
            "failed step became reusable or committed partial tokens");
    require(failed.cancel(1) && failed.releasable(), "failed work could not be released");

    // Requests that diverge together can form a child with inherited progress.
    Work trunk(make_prompt(4096, {10, 11}), 1);
    const auto child_a = make_prompt(4096, {20, 21});
    const auto child_b = make_prompt(4096, {20, 22});
    require(trunk.join(2, child_a).joined && trunk.join(3, child_b).joined,
            "nested branch setup failed");
    advance(trunk, 4096);
    Work child(child_a, 2, 8, trunk.completed());
    require(trunk.detach(2) && trunk.detach(3), "nested branches failed to detach");
    const auto nested = child.join(3, child_b);
    require(nested.joined && nested.common_tokens == 4097 && child.next_boundary() == 4097,
            "child work lost its additional common suffix");
    const auto child_step = advance(child, 1024);
    require(child_step.begin == 4096 && child_step.end == 4097 &&
                trunk.completed() == 4096 && child.at_boundary(),
            "nested branch restarted inherited work or changed its parent");
    require(child.detach(3) && advance(child, 1024).end == 4098,
            "nested child could not continue its private suffix");

    Work limited(make_prompt(10, {}), 1, 2);
    require(limited.join(2, make_prompt(10, {})).joined, "bounded join setup failed");
    const auto full = limited.join(3, make_prompt(10, {}));
    require(!full.joined && full.common_tokens == 10 && limited.dependents().size() == 2,
            "dependent limit allowed unbounded growth");
    require(rejects([&] { (void)limited.join(1, make_prompt(10, {})); }) &&
                rejects([&] { (void)limited.begin_step(0); }) &&
                rejects([&] { limited.complete_step(true); }) &&
                rejects([&] { Work invalid({}, 1); }) &&
                rejects([&] { Work invalid(make_prompt(0, {}), 1); }) &&
                rejects([&] { Work invalid(make_prompt(10, {}), 1, 0); }) &&
                rejects([&] { Work invalid(make_prompt(10, {}), 1, 2, 11); }),
            "invalid work state was accepted");
    require(limited.detach(2) && limited.join(3, make_prompt(10, {})).joined,
            "removed dependent did not return bounded capacity");
    Work distinct(make_prompt(10, {}), 1);
    const auto miss = distinct.join(2, make_prompt(0, {8}));
    require(!miss.joined && miss.common_tokens == 0 && distinct.dependents().size() == 1,
            "unrelated prompts joined shared work");

    const auto image = [](std::uint32_t begin, std::uint32_t end,
                          std::uint8_t identity) {
      prefix_index::ImageSpan result{begin, end};
      result.digest.fill(identity);
      return result;
    };
    const auto image_prompt = make_prompt(20, {});
    const std::vector<prefix_index::ImageSpan> images{
        image(4, 9, 1), image(12, 17, 2)};
    auto changed = images;
    changed[1].digest.fill(3);
    require(common_tokens(*image_prompt, images, *image_prompt, images) == 20 &&
                common_tokens(*image_prompt, images, *image_prompt, changed) == 12 &&
                common_tokens(*image_prompt, images, *image_prompt, {}) == 4 &&
                common_tokens(*image_prompt, {}, *image_prompt, images) == 4,
            "image identity lost the shared earlier image or matched plain text");
    std::swap(changed[0].digest, changed[1].digest);
    require(common_tokens(*image_prompt, images, *image_prompt, changed) == 4,
            "reordered image content shared feature tokens");
    changed = images;
    changed[0].end = 8;
    require(common_tokens(*image_prompt, images, *image_prompt, changed) == 4,
            "different image span lengths shared feature tokens");
    auto changed_tokens = *image_prompt;
    changed_tokens[7] = 8;
    require(common_tokens(*image_prompt, images, changed_tokens, images) == 4 &&
                common_tokens(*image_prompt, images,
                              std::vector<std::uint32_t>(7, 7), images) == 4 &&
                common_tokens(*image_prompt, images,
                              std::vector<std::uint32_t>(9, 7), images) == 9,
            "token mismatch or truncation split an image");
    changed_tokens = *image_prompt;
    changed_tokens[10] = 8;
    require(common_tokens(*image_prompt, images, changed_tokens, images) == 10,
            "text following an image rounded down a safe fork");
    auto zero_digest = images;
    zero_digest[0].digest.fill(0);
    require(common_tokens(*image_prompt, zero_digest, *image_prompt, {}) == 4,
            "zero image identity matched the text tag");

    // Changed later images share the first image and fork before the second.
    changed = images;
    changed[1].digest.fill(3);
    Work image_trunk(image_prompt, 1, 8, 0, images);
    const auto image_join = image_trunk.join(2, image_prompt, changed);
    require(image_join.joined && image_join.common_tokens == 12 &&
                advance(image_trunk, 1024).end == 4 &&
                advance(image_trunk, 1).end == 9 &&
                advance(image_trunk, 1024).end == 12 && image_trunk.at_boundary(),
            "image sharing split features or skipped the second-image fork");
    Work image_branch(image_prompt, 2, 8, image_trunk.completed(), changed);
    require(image_trunk.detach(2) && advance(image_branch, 1).end == 17 &&
                image_trunk.completed() == 12 && advance(image_trunk, 1).end == 17,
            "image branch replayed the common image or split its own features");

    // Work owns image identities after the original request is canceled. An
    // in-flight image remains indivisible even for a late join or tiny chunk.
    auto owned_images = std::vector<prefix_index::ImageSpan>{image(4, 1124, 1)};
    const auto original_images = owned_images;
    auto large_prompt = make_prompt(1130, {});
    const auto large_waiter = make_prompt(1128, {8});
    Work image_handoff(large_prompt, 1, 8, 0, owned_images);
    owned_images[0].digest.fill(9);
    require(image_handoff.images()[0].digest == original_images[0].digest &&
                advance(image_handoff, 16).end == 4,
            "work borrowed mutable image metadata or crossed feature begin");
    const auto large_step = image_handoff.begin_step(16);
    require(large_step.begin == 4 && large_step.end == 1124 &&
                image_handoff.join(2, large_waiter, original_images).joined,
            "large image split at the chunk cap or rejected a safe late join");
    const auto changed_late = image_handoff.join(3, large_waiter, owned_images);
    require(!changed_late.joined && changed_late.common_tokens == 4 &&
                image_handoff.join(4, make_prompt(1124, {}), original_images).joined &&
                image_handoff.cancel(1),
            "late image join rewound dispatch or lost its complete-image endpoint");
    large_prompt.reset();
    image_handoff.complete_step(true);
    require(image_handoff.completed() == 1124 && image_handoff.at_boundary() &&
                image_handoff.images()[0].digest == original_images[0].digest &&
                image_handoff.detach(4) && advance(image_handoff, 16).end == 1128,
            "representative cancellation lost image identity or restarted features");

    Work image_restore(image_prompt, 1, 8, 0, images);
    require(rejects([&] { image_restore.restore_prefix(6); }) &&
                image_restore.completed() == 0 &&
                rejects([&] { Work invalid(image_prompt, 1, 8, 6, images); }),
            "restored or inherited cursor split an image");
    image_restore.restore_prefix(4);
    require(advance(image_restore, 1).end == 9,
            "restored feature-begin cursor failed to dispatch the whole image");
    Work after_image(image_prompt, 1, 8, 9, images);
    require(advance(after_image, 1024).end == 12,
            "inherited image endpoint replayed features or crossed the next image");
    require(rejects([&] {
              Work invalid(image_prompt, 1, 8, 0,
                           {image(4, 9, 1), image(8, 12, 2)});
            }) && rejects([&] {
              Work invalid(image_prompt, 1, 8, 0, {image(4, 21, 1)});
            }) && rejects([&] {
              (void)image_restore.join(2, image_prompt, {image(4, 4, 1)});
            }), "invalid image spans were accepted");
    const std::vector<prefix_index::ImageSpan> leading_image{image(0, 4, 1)};
    Work leading(image_prompt, 1, 8, 0, leading_image);
    require(!leading.join(2, image_prompt).joined && advance(leading, 1).end == 4,
            "feature-begin zero matched plain text or split its first image");
    return true;
  } catch (const std::exception& error) {
    if (failure) *failure = error.what();
    return false;
  }
}

}  // namespace
}  // namespace gewell::pending_prefix

int main() {
  std::string failure;
  if (!gewell::pending_prefix::run_self_tests(&failure)) {
    std::cerr << failure << "\n";
    return 1;
  }
  std::cout << "pending_prefix tests: ok\n";
  return 0;
}
