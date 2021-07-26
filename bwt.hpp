#include <string>
#include <map>
#include <vector>

const char ETX = '$';

std::pair<std::string,int*> bwt_with_suffix_array(const std::string sequence);
std::string bwt(const std::string sequence);