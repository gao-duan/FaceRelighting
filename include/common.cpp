#include "common.h"
#include <fstream>
using namespace std;

bool file_exists(const std::string & name)
{
	ifstream in(name);
	bool ret;
	if (!in) {
		ret = false;
	}
	ret = true;
	in.close();
	return ret;
}

