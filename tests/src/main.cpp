// test/src/main.cpp
#include "common/tests_pch.h"
#include "ic/core/events.h"

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

	void onEvent(ic::Event& e) {
		(void)e;
	}
};

int main() {

	return 0;
}
