// test/src/main.cpp
#include "common/tests_pch.h"
#include "ic/core/events.h"
#include <spdlog/spdlog.h>

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


class String
{
public:
	String() = default;

	String(const char* string)
	{
		printf("Created\n");
		m_size = static_cast<uint32_t>(strlen(string));
		m_data = new char[m_size];
		memcpy(m_data, string, m_size);
	}

	String(const String& other)
	{
		printf("Copied!\n");
		m_size = other.m_size;
		m_data = new char[m_size];
		memcpy(m_data, other.m_data, m_size);
	}

	~String()
	{
		delete m_data;
	}

	void print()
	{
		for (size_t i{}; i < m_size; i++)
			spdlog::info("%c", m_data[i]);
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

	void printName()
	{
		m_name.print();
	}
private:
	String m_name;
};

int main() {



	return 0;
}
