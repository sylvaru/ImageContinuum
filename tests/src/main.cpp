
#include "image_continuum/core/app_base.h"

#include <chrono>
#include <iostream>

struct TestLayer 
{
	void onUpdate(float dt) {
		volatile float x = dt * 2.0f;
		(void)x;
	}
	void onRender(float alpha) {
		volatile float y = alpha + 1.0f;
		(void)y;
	}
};

int main() {

	ic::LayerStack stack;

	// Populate the stack with a massive amount of layers to get measurable timings
	constexpr int NUM_LAYERS = 10000;
	for (int i = 0; i < NUM_LAYERS; ++i) {
		stack.pushLayer(TestLayer{});
	}

	// Benchmark setup using high_resolution_clock
	constexpr int WARMUP_ITERATIONS = 100;
	constexpr int TIMED_ITERATIONS = 100000;

	// Warmup phase 
	// let CPU boost up clock speeds and ensure branch predictors settle
	for (int i = 0; i < WARMUP_ITERATIONS; ++i) {
		stack.updateAndRenderAll(0.016f, 1.0f);
	}

	// Timed phase
	auto start_time = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < TIMED_ITERATIONS; ++i) {
		stack.updateAndRenderAll(0.016f, 1.0f);
	}

	auto end_time = std::chrono::high_resolution_clock::now();

	// Calculate duration
	auto elapsed_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
	double avg_ns_per_frame = static_cast<double>(elapsed_ns) / TIMED_ITERATIONS;
	double avg_ms_per_frame = avg_ns_per_frame / 1e6;

	std::cout << "Executed " << TIMED_ITERATIONS << " frames over "
		<< NUM_LAYERS << " contiguous layers per frame." << std::endl;
	std::cout << "Total time: " << (elapsed_ns / 1e9) << " seconds." << std::endl;
	std::cout << "Average time per updateAndRenderAll call: " << avg_ns_per_frame << " ns (" << avg_ms_per_frame << " ms)" << std::endl;

}
