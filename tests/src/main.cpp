// test/src/main.cpp
#include "common/tests_pch.h"
#include "ic/core/events.h"
#include <spdlog/spdlog.h>

using namespace spdlog;



class String
{
public:
	String()
		: m_data(nullptr), m_size(0)
	{
	}

	String(const char* string)
	{
		info("Created\n");
		m_size = static_cast<uint32_t>(strlen(string));
		m_data = new char[m_size];
		memcpy(m_data, string, m_size);
	}

	String(const String& other)
	{
		info("Copied \n");
		m_size = other.m_size;

		if (m_size > 0)
		{
			m_data = new char[m_size];
			memcpy(m_data, other.m_data, m_size);
		}
		else
		{
			m_data = nullptr;
		}

	}

	String(String&& other) noexcept
	{
		info("Moved \n");
		m_size = other.m_size;
		m_data = other.m_data;

		other.m_size = 0;
		other.m_data = nullptr;
	}

	String& operator=(const String& other)
	{
		info("Copy assigned");

		if (this != &other)
		{
			delete[] m_data;

			m_size = other.m_size;

			if (m_size > 0)
			{
				m_data = new char[m_size];
				std::memcpy(m_data, other.m_data, m_size);
			}
			else
			{
				m_data = nullptr;
			}
		}

		return *this;
	}

	String& operator=(String&& other) noexcept
	{
		info("Move assigned \n");

		if (this != &other)
		{
			delete[] m_data;

			m_size = other.m_size;
			m_data = other.m_data;

			other.m_size = 0;
			other.m_data = nullptr;
		}

		return *this;


	}

	~String()
	{
		info("Destroyed");
		delete[] m_data;
	}

	void print()
	{
		for (size_t i{}; i < m_size; i++)
		{
			std::cout << m_data[i];
		}

		std::cout << "\n";
	}

private:
	char* m_data;
	uint32_t m_size;
};


class Entity
{
public:
	Entity(const String& name)
		: m_name(name)
	{

	}

	Entity(String&& name)
		: m_name(std::move(name))
	{
	}

	void printName()
	{
		m_name.print();
	}
private:
	String m_name;
};

int main() {

	Entity ent("Sylvaru");
	ent.printName();

	String apple = "Apple";

	String dest;

	apple.print();
	dest.print();

	dest = std::move(apple);

	apple.print();
	dest.print();

	std::cin.get();

	return 0;
}
