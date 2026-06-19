#pragma once
#include <cstdint>


namespace ic
{
	struct StateTransitionCommand
	{
		enum class Type 
		{ 
			PushLayer,
			PopLayer,
			ChangeScene 
		};
		uint64_t targetStateID;
	};
}