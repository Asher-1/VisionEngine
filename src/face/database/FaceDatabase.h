#pragma once

#include <map>
#include <memory>
#include <vector>

#include <opencv2/core.hpp>
#include "./stream/FileSystem.h"
#include "../../common/common.h"

namespace mirror {

class FaceDatabase {
public:
    FaceDatabase(const FaceDatabase &other) = delete;
    const FaceDatabase &operator=(const FaceDatabase &other) = delete;
	FaceDatabase();
	~FaceDatabase();

	void Clear();
	bool IsEmpty() const;
	int Load(const char* path);
	int Save(const char* path) const;
	int Delete(const std::string& name);
	int Find(std::vector<std::string>& names) const;
	int64_t Insert(const std::vector<float>& feat, const std::string& name);
	int QueryTop(const std::vector<float>& feat, QueryResult& query_result) const;


private:
	class Impl;
	Impl* impl_;

};
}