#include "XMLHelper.h"

using namespace tinyxml2;

void Tinyxml_Reader::Use(XMLElement* base) {
	basenode = base;
	if (!base) {
		printf("WARNING: NULL Node Being Used.\n");
	}
}

const char* Tinyxml_Reader::GetText(const char* name) {
	float tmp=0;
	if (basenode!=NULL) {
		XMLElement* attr = basenode->FirstChildElement(name);
		if (attr)
			return attr->GetText();
	}
	return NULL;
}

float Tinyxml_Reader::GetFloat(const char* name) {
	float tmp=0;
	if (basenode!=NULL) {
		XMLElement* attr = basenode->FirstChildElement(name);
		if (attr)
			attr->QueryFloatText(&tmp);
	}
	return tmp;
}

int Tinyxml_Reader::GetInt(const char* name) {
	int tmp=0;
	if (basenode!=NULL) {
		basenode->FirstChildElement(name)->QueryIntText(&tmp);
	}
	return tmp;
}