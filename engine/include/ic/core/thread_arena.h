#pragma once 


namespace ic
{
	struct ThreadArena
	{
		uint8_t* begin;
		uint8_t* current;
		uint8_t* end;
	}
}