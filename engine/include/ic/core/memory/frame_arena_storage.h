#pragma once


namespace ic
{
	class FrameArenaStorage
	{
	public:
        FrameArenaStorage(
            size_t capacity)
            : m_memory(std::make_unique<std::byte[]>(capacity))
            , m_capacity(capacity)
        {
        }

        void* base()
        {
            return m_memory.get();
        }

        size_t capacity() const
        {
            return m_capacity;
        }

	private:

		std::unique_ptr<std::byte[]> m_memory;
		size_t m_capacity;
	};
}