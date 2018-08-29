#pragma once
#include "../include/tinyxml2.h"

using namespace tinyxml2;

struct Tinyxml_Reader {
	XMLElement* basenode = NULL;

	void Use(XMLElement* base);
	const char* GetText(const char* name);
	float GetFloat(const char* name);
	int GetInt(const char* name);
};