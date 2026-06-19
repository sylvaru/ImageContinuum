// layer.h
#pragma once

namespace ic
{
	template <typename Derived>
	class Layer
	{
	public:
		void dispatchUpdate(float dt)
		{
			// Casting through void* bypasses strict incomplete type checks 
			// while maintaining zero runtime overhead.
			static_cast<Derived*>(static_cast<void*>(this))->onUpdate(dt);
		}

		void dispatchRender(float alpha)
		{
			static_cast<Derived*>(static_cast<void*>(this))->onRender(alpha);
		}

	};
}