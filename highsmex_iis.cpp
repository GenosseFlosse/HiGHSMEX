/* MATLAB mex wrapper over the irreducible infeasible subsystem (IIS) facility of
 * the HiGHS optimization library (https://github.com/ERGO-Code/HiGHS)
 *
 * This is a stand-alone companion to highsmex.cpp. It builds an LP from the same
 * leading input arguments as highsmex (c, A, L, U, l, u) and calls
 * Highs::getIis(HighsIis&) instead of Highs::run().
 *
 * The HiGHS IIS facility is documented at
 * https://ergo-code.github.io/HiGHS/stable/guide/advanced/
 * and is currently available for LPs only (no MIP, no QP).
 *
 * Covered by the MIT License (see LICENSE file for details).
 * See https://github.com/savyasachi/HiGHSMEX for more information.
 */


 // Include standard headers
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stack>
#include <sstream>
#include <functional>
#include <exception>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <utility>
#include <complex>
// Include C++ MEX API
#include "mex.hpp"
#include "mexAdapter.hpp"
#include "MatlabDataArray.hpp"
// Include HiGHS. Highs.h pulls in lp_data/HighsIis.h, which defines HighsIis,
// HighsIisInfo, IisBoundStatus and IisModelStatus. The IisStrategy and IisStatus
// enumerations come from lp_data/HConst.h.
#include "Highs.h"


// Open MATLAB namespaces
using namespace matlab::mex;
using namespace matlab::data;
using namespace matlab::engine;


/* ------------------------------------------------------------------------------------------------------ */
/*                                        ENUMERATIONS                                                    */
/* ------------------------------------------------------------------------------------------------------ */

enum class MexCallSyntax { kVer, kDefaultOpts, kIntType, kIis };


/* ------------------------------------------------------------------------------------------------------ */
/*                                         VARIABLES                                                      */
/* ------------------------------------------------------------------------------------------------------ */

// Compile time constants
constexpr ArrayType   HighsInt2MatlabArrayType = std::is_same_v<HighsInt, int32_t> ? ArrayType::INT32 : ArrayType::INT64;
const std::string HighsInt2MatlabClassStr = std::is_same_v<HighsInt, int32_t> ? "int32" : "int64";
constexpr bool        MexDebugPrinting = 0;

/* Default value used for the HiGHS option "iis_strategy".
 * The HiGHS default is kIisStrategyLight (0), which only performs the cheap
 * trivial checks (inconsistent bounds, empty infeasible rows, row value bounds)
 * and usually returns an empty IIS. Since the whole point of this mex function
 * is to determine an IIS, the default here is
 *   kIisStrategyFromLp (2)      - build a mutually infeasible set of rows using
 *                                 the elasticity filter, and
 *   kIisStrategyIrreducible (4) - reduce that set to an irreducible one.
 * Pass an "iis_strategy" field in the options struct to override this.
 */
constexpr HighsInt kHighsMexDefaultIisStrategy = kIisStrategyFromLp | kIisStrategyIrreducible;

// Maps of the HiGHS IIS enumerations onto the strings reported to MATLAB
const std::map<HighsInt, std::string> iisBoundStatusToStringMap({
	{kIisBoundStatusDropped, "dropped"},
	{kIisBoundStatusNull,    "null"},
	{kIisBoundStatusFree,    "free"},
	{kIisBoundStatusLower,   "lower"},
	{kIisBoundStatusUpper,   "upper"},
	{kIisBoundStatusBoxed,   "boxed"}
	});
const std::map<HighsInt, std::string> iisStatusToStringMap({
	{kIisStatusNotInConflict,   "notInConflict"},
	{kIisStatusMaybeInConflict, "maybeInConflict"},
	{kIisStatusInConflict,      "inConflict"}
	});
const std::map<HighsInt, std::string> iisModelStatusToStringMap({
	{kIisModelStatusFeasible,    "feasible"},
	{kIisModelStatusUnknown,     "unknown"},
	{kIisModelStatusTimeLimit,   "timeLimit"},
	{kIisModelStatusReducible,   "reducible"},
	{kIisModelStatusIrreducible, "irreducible"}
	});

// Variables
ArrayFactory factory;


/* ------------------------------------------------------------------------------------------------------ */
/*                                          FUNCTIONS                                                     */
/* ------------------------------------------------------------------------------------------------------ */

template <typename T>
constexpr HighsInt castToHighsInt(const T x) {
	return static_cast<HighsInt>(x);
}


// Returns the type of the MATLAB array.
inline ArrayType getType(const Array& arr) noexcept(noexcept(arr.getType())) {
	return arr.getType();
}


// Returns true if the input MATLAB array (non-sparse) is double type.
inline bool isDouble(const Array& arr) noexcept(noexcept(getType(arr))) {
	return getType(arr) == ArrayType::DOUBLE;
}


// Returns true if the input MATLAB array is a struct.
inline bool isStruct(const Array& arr) noexcept(noexcept(getType(arr))) {
	return getType(arr) == ArrayType::STRUCT;
}


// Returns true if the input MATLAB array is a cell.
inline bool isCell(const Array& arr) noexcept(noexcept(getType(arr))) {
	return getType(arr) == ArrayType::CELL;
}


// Returns true if the input MATLAB array is a MATLAB string (delimited by double quotes "").
inline bool isMatlabString(const Array& arr) noexcept(noexcept(getType(arr))) {
	return getType(arr) == ArrayType::MATLAB_STRING;
}


// Returns number of elements in a matlab::data::Array.
inline size_t numel(const Array& arr) noexcept(noexcept(arr.getNumberOfElements())) {
	return arr.getNumberOfElements();
}


// Returns true if the input matlab::data::Array is empty.
inline bool isEmpty(const Array& arr) noexcept(noexcept(arr.isEmpty())) {
	return arr.isEmpty();
}


// Returns true if the input matlab::data::Array has size 1.
inline bool isScalar(const Array& arr) noexcept(noexcept(numel(arr))) {
	return numel(arr) == 1;
}


// Returns true if the input ArrayDimensions is a vector.
inline bool isVector(const ArrayDimensions& dims) {
	return dims.size() == 2 && ((dims[0] == 1 && dims[1] > 0) || (dims[0] > 0 && dims[1] == 1));
}


// Returns true if the input matlab::data::Array is a vector.
inline bool isVectorArr(const Array& arr) {
	return isVector(arr.getDimensions());
}


// Returns true if the input ArrayDimensions is a matrix.
inline bool isMatrix(const ArrayDimensions& dims) {
	return dims.size() == 2 && dims[0] > 0 && dims[1] > 0;
}


// Get the names of fields of a struct array.
inline std::vector<std::string> getFieldNames(const StructArray& arr) {
	auto frange = arr.getFieldNames();
	return { frange.begin(), frange.end() };
}


// Convert MATLAB string to std::string.
inline std::string matlabStringToStdString(const MATLABString& matlabStr) {
	return matlabStr.has_value() ? convertUTF16StringToUTF8String(*matlabStr) : "";
}

template <typename T>
void appendToStream(std::ostringstream& oss, const T& value) {
	oss << value;
}

inline void appendToStream(std::ostringstream& oss, const std::string& value) {
	oss << value;
}

inline void appendToStream(std::ostringstream& oss, const char* value) {
	oss << value;
}

inline void appendToStream(std::ostringstream& oss, const MATLABString& value) {
	oss << matlabStringToStdString(value);
}

template <typename... Args>
std::string formatMessage(const Args&... args) {
	std::ostringstream oss;
	(appendToStream(oss, args), ...);
	return oss.str();
}


template <typename>
struct is_std_complex : public std::false_type {};

template <typename T>
struct is_std_complex<std::complex<T>> : public std::true_type {};

// Extracts the pointer to underlying data from the non-const iterator (`TypedIterator<T>`).
/* This function does not throw any exceptions. */
template <typename T>
inline T* toPointer(const matlab::data::TypedIterator<T>& it) noexcept(noexcept(it.operator->())) {
	static_assert((std::is_arithmetic<T>::value || is_std_complex<T>::value) && !std::is_const<T>::value,
		"Template argument T must be a std::is_arithmetic type or std::complex.");
	return it.operator->();
}


/* Extracts pointer to the first element in the array.
 * NOTE: Do not call `getPointer` with a temporary object.
 */
template <typename T>
inline T* getPointer(matlab::data::TypedArray<T>& arr) noexcept(noexcept(toPointer(arr.begin()))) {
	return toPointer(arr.begin());
}
template <typename T>
inline const T* getPointer(const matlab::data::TypedArray<T>& arr) noexcept(noexcept(toPointer(arr.begin()))) {
	return getPointer(const_cast<matlab::data::TypedArray<T>&>(arr));
}


// Convert std::vector to MATLAB vector.
template <typename T>
TypedArray<T> stdVectorToMatlabVector(const std::vector<T>& v, const bool rowShape) {
	auto out = factory.createArray<T>(rowShape ? ArrayDimensions({ 1, v.size() }) : ArrayDimensions({ v.size(), 1 }));
	std::copy(v.begin(), v.end(), getPointer(out));
	return out;
}


// Convert MATLAB vector to std::vector.
template <typename T>
inline std::vector<T> matlabVectorToStdVector(const TypedArray<T>& arr) {
	auto pBegin = getPointer(arr);
	return { pBegin, pBegin + numel(arr) };
}


// Pre-condition: indx < numel(matStruct)
inline void throwIfInvalidFieldValue(const StructArray& matStruct, const size_t indx, const std::string& fieldname, const ArrayType fieldType,
	std::function<bool(const Array& arr)> fieldValueCheck, const std::string& errMsg) {
	if (!(getType(matStruct[indx][fieldname]) == fieldType && fieldValueCheck(matStruct[indx][fieldname]))) {
		throw std::runtime_error(errMsg);
	}
}


/* Convert a vector of zero based HiGHS indices to a MATLAB (one based) column
 * vector of doubles, so that the result can be used for indexing straight away.
 */
TypedArray<double> highsIndexVectorToMatlabVector(const std::vector<HighsInt>& v) {
	auto out = factory.createArray<double>(ArrayDimensions({ v.size(), 1 }));
	auto p = getPointer(out);
	for (size_t i = 0; i < v.size(); ++i) {
		p[i] = static_cast<double>(v[i]) + 1; // +1 to convert C++ (zero) based index to MATLAB (one) based index
	}
	return out;
}


// Convert a vector of HiGHS enumeration values to a MATLAB column vector of strings.
TypedArray<MATLABString> highsIntVectorToMatlabStringVector(const std::vector<HighsInt>& v, const std::map<HighsInt, std::string>& valueToStringMap) {
	auto out = factory.createArray<MATLABString>(ArrayDimensions({ v.size(), 1 }));
	std::transform(v.begin(), v.end(), out.begin(),
		[&valueToStringMap](const HighsInt value_) -> std::string {
			auto const it_ = valueToStringMap.find(value_);
			return it_ == valueToStringMap.end() ? std::string("unknown") : it_->second;
		});
	return out;
}


// Look up an enumeration value in one of the maps above, without throwing for unknown values.
std::string highsIntToString(const HighsInt value, const std::map<HighsInt, std::string>& valueToStringMap) {
	auto const it = valueToStringMap.find(value);
	return it == valueToStringMap.end() ? std::string("unknown") : it->second;
}


/* Convert a HighsSparseMatrix to a MATLAB cell array in the {i, j, v, nr, nc}
 * format accepted by callhighs, where [i, j, v]=find(A) and [nr, nc]=size(A).
 * The row and column indices are one based. Both the col-wise and the row-wise
 * HiGHS matrix formats are handled.
 */
CellArray highsSparseMatrixToMatlabCell(const HighsSparseMatrix& matrix) {
	const HighsInt numRow = matrix.num_row_;
	const HighsInt numCol = matrix.num_col_;
	const bool rowwise = matrix.format_ == MatrixFormat::kRowwise;
	// A matrix with no entries may carry an empty start_ vector, in which case
	// there is nothing to iterate over.
	const HighsInt numVec = matrix.start_.empty()
		? 0
		: std::min(rowwise ? numRow : numCol, castToHighsInt(matrix.start_.size()) - 1);
	const size_t nnz = numVec > 0 ? static_cast<size_t>(matrix.start_[numVec]) : 0;

	auto iArr = factory.createArray<double>(ArrayDimensions({ nnz, 1 }));
	auto jArr = factory.createArray<double>(ArrayDimensions({ nnz, 1 }));
	auto vArr = factory.createArray<double>(ArrayDimensions({ nnz, 1 }));
	if (nnz) {
		auto pI = getPointer(iArr), pJ = getPointer(jArr), pV = getPointer(vArr);
		size_t k = 0;
		for (HighsInt iVec = 0; iVec < numVec; ++iVec) {
			for (HighsInt iEl = matrix.start_[iVec]; iEl < matrix.start_[iVec + 1]; ++iEl) {
				const HighsInt indx = matrix.index_[iEl];
				// +1 to convert C++ (zero) based indices to MATLAB (one) based indices
				pI[k] = static_cast<double>(rowwise ? iVec : indx) + 1;
				pJ[k] = static_cast<double>(rowwise ? indx : iVec) + 1;
				pV[k] = matrix.value_[iEl];
				++k;
			}
		}
	}
	return factory.createCellArray({ 1, 5 },
		std::move(iArr), std::move(jArr), std::move(vArr),
		factory.createScalar<double>(static_cast<double>(numRow)),
		factory.createScalar<double>(static_cast<double>(numCol)));
}


// Convert HighsIisInfo to a MATLAB struct.
StructArray highsIisInfoToMatlabStruct(const HighsIisInfo& info) {
	auto out = factory.createStructArray({ 1, 1 },
		{ "num_lp_solved", "sum_simplex_iteration_counts", "min_simplex_iteration_count",
		  "max_simplex_iteration_count", "sum_simplex_times", "min_simplex_time", "max_simplex_time" });
	out[0]["num_lp_solved"] = factory.createScalar(info.num_lp_solved);
	out[0]["sum_simplex_iteration_counts"] = factory.createScalar(info.sum_simplex_iteration_counts);
	out[0]["min_simplex_iteration_count"] = factory.createScalar(info.min_simplex_iteration_count);
	out[0]["max_simplex_iteration_count"] = factory.createScalar(info.max_simplex_iteration_count);
	out[0]["sum_simplex_times"] = factory.createScalar(info.sum_simplex_times);
	out[0]["min_simplex_time"] = factory.createScalar(info.min_simplex_time);
	out[0]["max_simplex_time"] = factory.createScalar(info.max_simplex_time);
	return out;
}


/* Convert HighsIis to a MATLAB struct. The fields mirror the data members of the
 * HighsIis class, with the indices converted to MATLAB (one based) indices and
 * the enumerations converted to strings.
 */
StructArray highsIisToMatlabStruct(const HighsIis& iis) {
	auto out = factory.createStructArray({ 1, 1 },
		{ // These fields mirror the data members of the HighsIis class of the HiGHS library
			"valid", "status", "strategy", "col_index", "row_index", "col_bound", "row_bound",
			"col_status", "row_status", "info",
			// These fields are extra. They are added by highsmex_iis.
			"status_string" });
	out[0]["valid"] = factory.createScalar(iis.valid_);
	out[0]["status"] = factory.createScalar(iis.status_);
	out[0]["strategy"] = factory.createScalar(iis.strategy_);
	out[0]["col_index"] = highsIndexVectorToMatlabVector(iis.col_index_);
	out[0]["row_index"] = highsIndexVectorToMatlabVector(iis.row_index_);
	out[0]["col_bound"] = highsIntVectorToMatlabStringVector(iis.col_bound_, iisBoundStatusToStringMap);
	out[0]["row_bound"] = highsIntVectorToMatlabStringVector(iis.row_bound_, iisBoundStatusToStringMap);
	out[0]["col_status"] = highsIntVectorToMatlabStringVector(iis.col_status_, iisStatusToStringMap);
	out[0]["row_status"] = highsIntVectorToMatlabStringVector(iis.row_status_, iisStatusToStringMap);
	out[0]["info"] = highsIisInfoToMatlabStruct(iis.info_);
	// Extra fields
	out[0]["status_string"] = factory.createScalar(highsIntToString(iis.status_, iisModelStatusToStringMap));
	return out;
}


/* Convert the LP held by a HighsIis (the infeasible subsystem itself) to a MATLAB
 * struct laid out like the input arguments of callhighs, so that it can be passed
 * straight back to callhighs / callhighs_iis.
 *
 * NOTE: HiGHS sets the column costs of the IIS LP to zero, since costs play no
 *       role in infeasibility. The bounds are the ones that characterise the IIS,
 *       i.e. bounds that were dropped during the reduction are infinite here.
 */
StructArray highsIisLpToMatlabStruct(const HighsIis& iis) {
	auto const& lp = iis.model_.lp_;
	auto out = factory.createStructArray({ 1, 1 },
		{ "c", "A", "L", "U", "l", "u", "colIndex", "rowIndex" });
	out[0]["c"] = stdVectorToMatlabVector(lp.col_cost_, false);
	out[0]["A"] = highsSparseMatrixToMatlabCell(lp.a_matrix_);
	out[0]["L"] = stdVectorToMatlabVector(lp.row_lower_, false);
	out[0]["U"] = stdVectorToMatlabVector(lp.row_upper_, false);
	out[0]["l"] = stdVectorToMatlabVector(lp.col_lower_, false);
	out[0]["u"] = stdVectorToMatlabVector(lp.col_upper_, false);
	out[0]["colIndex"] = highsIndexVectorToMatlabVector(iis.col_index_);
	out[0]["rowIndex"] = highsIndexVectorToMatlabVector(iis.row_index_);
	return out;
}


// Convert HighsInfo to MATLAB struct. We add some extra fields to the output MATLAB struct.
StructArray highsInfoToMatlabStruct(const Highs& highs) {
	auto out = factory.createStructArray({ 1, 1 },
		{ // These fields mirror the HighsInfo class of the HiGHS library
			"valid", "mip_node_count", "simplex_iteration_count", "ipm_iteration_count", "crossover_iteration_count",
		"pdlp_iteration_count", "qp_iteration_count", "primal_solution_status", "dual_solution_status", "basis_validity",
		"objective_function_value", "mip_dual_bound", "mip_gap", "max_integrality_violation", "num_primal_infeasibilities",
		"max_primal_infeasibility", "sum_primal_infeasibilities", "num_dual_infeasibilities", "max_dual_infeasibility",
		"sum_dual_infeasibilities", "num_relative_primal_infeasibilities", "max_relative_primal_infeasibility",
		"num_relative_dual_infeasibilities", "max_relative_dual_infeasibility", "num_primal_residual_errors",
		"max_primal_residual_error", "num_dual_residual_errors", "max_dual_residual_error",
		"num_relative_primal_residual_errors", "max_relative_primal_residual_error", "num_relative_dual_residual_errors",
		"max_relative_dual_residual_error", "num_complementarity_violations", "max_complementarity_violation",
		"primal_dual_objective_error", "primal_dual_integral",
		// These fields are extra. They are added by highsmex_iis.
		"primal_solution_status_string", "dual_solution_status_string", "basis_validity_string", "model_status_string", "run_time" });
	auto const& info = highs.getInfo();
	out[0]["valid"] = factory.createScalar(info.valid);
	out[0]["mip_node_count"] = factory.createScalar(info.mip_node_count);
	out[0]["simplex_iteration_count"] = factory.createScalar(info.simplex_iteration_count);
	out[0]["ipm_iteration_count"] = factory.createScalar(info.ipm_iteration_count);
	out[0]["crossover_iteration_count"] = factory.createScalar(info.crossover_iteration_count);
	out[0]["pdlp_iteration_count"] = factory.createScalar(info.pdlp_iteration_count);
	out[0]["qp_iteration_count"] = factory.createScalar(info.qp_iteration_count);
	out[0]["primal_solution_status"] = factory.createScalar(info.primal_solution_status);
	out[0]["dual_solution_status"] = factory.createScalar(info.dual_solution_status);
	out[0]["basis_validity"] = factory.createScalar(info.basis_validity);
	out[0]["objective_function_value"] = factory.createScalar(info.objective_function_value);
	out[0]["mip_dual_bound"] = factory.createScalar(info.mip_dual_bound);
	out[0]["mip_gap"] = factory.createScalar(info.mip_gap);
	out[0]["max_integrality_violation"] = factory.createScalar(info.max_integrality_violation);
	out[0]["num_primal_infeasibilities"] = factory.createScalar(info.num_primal_infeasibilities);
	out[0]["max_primal_infeasibility"] = factory.createScalar(info.max_primal_infeasibility);
	out[0]["sum_primal_infeasibilities"] = factory.createScalar(info.sum_primal_infeasibilities);
	out[0]["num_dual_infeasibilities"] = factory.createScalar(info.num_dual_infeasibilities);
	out[0]["max_dual_infeasibility"] = factory.createScalar(info.max_dual_infeasibility);
	out[0]["sum_dual_infeasibilities"] = factory.createScalar(info.sum_dual_infeasibilities);
	out[0]["num_relative_primal_infeasibilities"] = factory.createScalar(info.num_relative_primal_infeasibilities);
	out[0]["max_relative_primal_infeasibility"] = factory.createScalar(info.max_relative_primal_infeasibility);
	out[0]["num_relative_dual_infeasibilities"] = factory.createScalar(info.num_relative_dual_infeasibilities);
	out[0]["max_relative_dual_infeasibility"] = factory.createScalar(info.max_relative_dual_infeasibility);
	out[0]["num_primal_residual_errors"] = factory.createScalar(info.num_primal_residual_errors);
	out[0]["max_primal_residual_error"] = factory.createScalar(info.max_primal_residual_error);
	out[0]["num_dual_residual_errors"] = factory.createScalar(info.num_dual_residual_errors);
	out[0]["max_dual_residual_error"] = factory.createScalar(info.max_dual_residual_error);
	out[0]["num_relative_primal_residual_errors"] = factory.createScalar(info.num_relative_primal_residual_errors);
	out[0]["max_relative_primal_residual_error"] = factory.createScalar(info.max_relative_primal_residual_error);
	out[0]["num_relative_dual_residual_errors"] = factory.createScalar(info.num_relative_dual_residual_errors);
	out[0]["max_relative_dual_residual_error"] = factory.createScalar(info.max_relative_dual_residual_error);
	out[0]["num_complementarity_violations"] = factory.createScalar(info.num_complementarity_violations);
	out[0]["max_complementarity_violation"] = factory.createScalar(info.max_complementarity_violation);
	out[0]["primal_dual_objective_error"] = factory.createScalar(info.primal_dual_objective_error);
	out[0]["primal_dual_integral"] = factory.createScalar(info.primal_dual_integral);
	// Extra fields
	out[0]["primal_solution_status_string"] = factory.createScalar(highs.solutionStatusToString(info.primal_solution_status));
	out[0]["dual_solution_status_string"] = factory.createScalar(highs.solutionStatusToString(info.dual_solution_status));
	out[0]["basis_validity_string"] = factory.createScalar(highs.basisValidityToString(info.basis_validity));
	out[0]["model_status_string"] = factory.createScalar(highs.modelStatusToString(highs.getModelStatus()));
	out[0]["run_time"] = factory.createScalar(highs.getRunTime());

	return out;
}


// Convert the HiGHS options to a MATLAB struct.
StructArray highsOptionsToMatlabStruct(const Highs& highs, const bool getDefaults) {
	// Get the names of all the user settable options
	auto const numOptions = highs.getNumOptions();
	std::vector<std::string> fieldnames(numOptions);
	for (HighsInt i = 0; i < numOptions; ++i) {
		highs.getOptionName(i, &fieldnames[i]); // Note: This will always return kOk here
	}
	auto out = factory.createStructArray({ 1, 1 }, fieldnames);
	// Get values of all the user settable options
	for (auto const& fn : fieldnames) {
		HighsOptionType optType;
		highs.getOptionType(fn, &optType); // Note: This will always return kOk here
		switch (optType) {
		case HighsOptionType::kBool:
		{
			bool value;
			if (getDefaults) {
				highs.getBoolOptionValues(fn, nullptr, &value);
			}
			else {
				highs.getBoolOptionValues(fn, &value, nullptr);
			}
			out[0][fn] = factory.createScalar(value);
			break;
		}

		case HighsOptionType::kInt:
		{
			HighsInt value;
			if (getDefaults) {
				highs.getIntOptionValues(fn, nullptr, nullptr, nullptr, &value);
			}
			else {
				highs.getIntOptionValues(fn, &value, nullptr, nullptr, nullptr);
			}
			out[0][fn] = factory.createScalar(value);
			break;
		}

		case HighsOptionType::kDouble:
		{
			double value;
			if (getDefaults) {
				highs.getDoubleOptionValues(fn, nullptr, nullptr, nullptr, &value);
			}
			else {
				highs.getDoubleOptionValues(fn, &value, nullptr, nullptr, nullptr);
			}
			out[0][fn] = factory.createScalar(value);
			break;
		}

		case HighsOptionType::kString:
		{
			std::string value;
			if (getDefaults) {
				highs.getStringOptionValues(fn, nullptr, &value);
			}
			else {
				highs.getStringOptionValues(fn, &value, nullptr);
			}
			out[0][fn] = factory.createScalar(value);
			break;
		}
		}
	}
	return out;
}


// Set HiGHS options by taking values from a MATLAB struct
// Pre-condition: opts is a 1x1 struct
void setHighsOptions(Highs& highs, const StructArray& opts, const std::string& mexArgInNumberAsStr) {
	// Get the fieldnames of the input MATLAB struct
	auto const fieldnames = getFieldNames(opts);
	// Set HiGHS options
	for (auto const& fn : fieldnames) {
		HighsOptionType optType;
		if (highs.getOptionType(fn, &optType) != HighsStatus::kOk) {
			throw std::runtime_error(formatMessage("Invalid option provided in the struct passed as the ", mexArgInNumberAsStr, " input argument. \"", fn, "\" is not a legal HiGHS option."));
		}
		switch (optType) {
		case HighsOptionType::kBool:
		{
			throwIfInvalidFieldValue(opts, 0, fn, ArrayType::LOGICAL, isScalar,
				formatMessage("Field \"", fn, "\" of the options struct passed as the ", mexArgInNumberAsStr, " input argument must be a scalar of logical type."));
			const TypedArray<bool> value = opts[0][fn];
			if (highs.setOptionValue(fn, static_cast<bool>(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(formatMessage("Failed to set the HiGHS option \"", fn, "\". The option struct was passed as the ", mexArgInNumberAsStr, " input argument."));
			}
			break;
		}

		case HighsOptionType::kInt:
		{
			throwIfInvalidFieldValue(opts, 0, fn, HighsInt2MatlabArrayType, isScalar,
				formatMessage("Field \"", fn, "\" of the options struct passed as the ", mexArgInNumberAsStr, " input argument must be a scalar of ", HighsInt2MatlabClassStr, " type."));
			const TypedArray<HighsInt> value = opts[0][fn];
			if (highs.setOptionValue(fn, castToHighsInt(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(formatMessage("Failed to set the HiGHS option \"", fn, "\". The option struct was passed as the ", mexArgInNumberAsStr, " input argument."));
			}
			break;
		}

		case HighsOptionType::kDouble:
		{
			throwIfInvalidFieldValue(opts, 0, fn, ArrayType::DOUBLE, isScalar,
				formatMessage("Field \"", fn, "\" of the options struct passed as the ", mexArgInNumberAsStr, " input argument must be a scalar of double type."));
			const TypedArray<double> value = opts[0][fn];
			if (highs.setOptionValue(fn, static_cast<double>(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(formatMessage("Failed to set the HiGHS option \"", fn, "\". The option struct was passed as the ", mexArgInNumberAsStr, " input argument."));
			}
			break;
		}

		case HighsOptionType::kString:
		{
			throwIfInvalidFieldValue(opts, 0, fn, ArrayType::MATLAB_STRING, isScalar,
				formatMessage("Field \"", fn, "\" of the options struct passed as the ", mexArgInNumberAsStr, " input argument must be a MATLAB string."));
			const TypedArray<MATLABString> value = opts[0][fn];
			if (highs.setOptionValue(fn, matlabStringToStdString(value[0])) != HighsStatus::kOk) {
				throw std::runtime_error(formatMessage("Failed to set the HiGHS option \"", fn, "\". The option struct was passed as the ", mexArgInNumberAsStr, " input argument."));
			}
			break;
		}
		}
	}
}


// Convert MATLAB full matrix to HIGHS sparse (col-wise) representation.
HighsInt matlabMatrixToHighsFormat(
	std::vector<HighsInt>& start, std::vector<HighsInt>& index, std::vector<double>& value, // outputs
	const TypedArray<double>& A, const HighsInt nrow, const HighsInt ncol // inputs
) {
	if (A.getMemoryLayout() != MemoryLayout::COLUMN_MAJOR) {
		throw std::runtime_error("Input matrix must be in column major order.");
	}
	auto pA = getPointer(A);
	// Count the number of non-zero elements
	auto pAcol = pA; // Pointer to the first element of the first column of A
	HighsInt nnz = 0;
	for (HighsInt j = 0; j < ncol; ++j) {
		for (HighsInt i = 0; i < nrow; ++i) {
			if (pAcol[i] != 0) ++nnz;
		}
		pAcol += nrow; // Move to the next column
	}
	// Resize outputs
	start.resize(ncol + 1);
	index.resize(nnz);
	value.resize(nnz);
	if (!nnz) {
		std::fill(start.begin(), start.end(), 0);
	}
	else {
		pAcol = pA;
		HighsInt k = 0;
		for (HighsInt j = 0; j < ncol; ++j) {
			start[j] = k;
			for (HighsInt i = 0; i < nrow; ++i) {
				if (!pAcol[i]) continue;
				index[k] = i;
				value[k] = pAcol[i];
				++k;
			}
			pAcol += nrow; // Move to the next column
		}
		// Here k == nnz
		start[ncol] = nnz;
	}
	return nnz;
}


/* Convert MATLAB sparse matrix to HIGHS sparse (col-wise) representation. The sparse
 * matrix must be specified by the triplet iA, jA, and, vA, where, [iA, jA, vA]=find(A).
 * Pre-condition: iA, jA, vA are vectors of the same length holding the row indices,
 * column indices (MATLAB based i.e. starting at 1) and values of the non-zero elements
 * of A, sorted in column major order (as returned by MATLAB's find).
 */
HighsInt matlabMatrixToHighsFormat(
	std::vector<HighsInt>& start, std::vector<HighsInt>& index, std::vector<double>& value, // outputs
	const TypedArray<double>& iA, const TypedArray<double>& jA, const TypedArray<double>& vA, const HighsInt, const HighsInt ncol // inputs
) {
	auto pI = getPointer(iA), pJ = getPointer(jA), pV = getPointer(vA);
	const size_t nA = numel(iA);
	const HighsInt nnz = castToHighsInt(nA);
	// Resize outputs
	start.resize(ncol + 1);
	index.resize(nnz);
	value.resize(nnz);
	// Loop over all the non-zero elements and copy the values to the outputs
	std::vector<HighsInt> nnzCol(ncol); // nnzCol[i] is the number of non-zero elements in the i'th column of A
	nnzCol.assign(ncol, 0);
	for (size_t i = 0; i < nA; ++i) {
		++nnzCol[castToHighsInt(pJ[i] - 1)]; // -1 to convert MATLAB (one) based index to C++ (zero) based index
		index[i] = castToHighsInt(pI[i] - 1); // -1 to convert MATLAB (one) based index to C++ (zero) based index
		value[i] = pV[i];
	}
	// Set start
	start[0] = 0;
	for (HighsInt j = 1; j <= ncol; ++j) {
		start[j] = start[j - 1] + nnzCol[j - 1];
	}
	return nnz;
}


/* ------------------------------------------------------------------------------------------------------ */
/*                                         MEX INTERFACE                                                  */
/* ------------------------------------------------------------------------------------------------------ */

class MexFunction : public Function {

	std::shared_ptr<MATLABEngine> mtlbEngPtr = getEngine();
	// Error messages passed by HiGHS inside the logging callback are stored here
	std::stack<std::string> highsLogErrStack;


	std::string getFunctionNameString() {
		return convertUTF16StringToUTF8String(getFunctionName());
	}

	// Display std::vector on the MATLAB console. This method is used while debugging.
	template <typename T>
	void disp(const std::vector<T>& v) {
		mtlbEngPtr->feval(u"disp", 0, std::vector<Array>({ stdVectorToMatlabVector(v, true) }));
	}

	// Display Array on the MATLAB console. This method is used while debugging.
	void disp(const Array& arr) {
		mtlbEngPtr->feval(u"disp", 0, std::vector<Array>({ arr }));
	}

	// Display string on the MATLAB console.
	void print(const std::string& msg) {
		mtlbEngPtr->feval(u"fprintf", 0, std::vector<Array>(
			{ factory.createScalar("%s"), factory.createScalar(msg) }));
	}

	// Display warning message.
	void warning(const std::string& msg) {
		auto const str = formatMessage("in mex function ", getFunctionNameString(), ": ", msg, "\n");
		mtlbEngPtr->feval(u"warning", 0, std::vector<Array>({
			factory.createScalar("highs:mex"), factory.createScalar(str)
			}));
	}

	// Display error message. This method is meant to be called inside the catch blocks of operator()(...) method.
	void error__(const std::string& msg) {
		auto const str = formatMessage("Error in mex function ", getFunctionNameString(), ":\n", msg, "\n");
		mtlbEngPtr->feval(u"error", 0, std::vector<Array>({ factory.createScalar(str) }));
	}

	// Callback to log HiGHS messages to the MATLAB console
	// Pre-condition: callbackType should always be HighsCallbackType::kCallbackLogging.
	void logCallback(const int, const HighsLogType logType, const std::string& message) {
		switch (logType) {
		case HighsLogType::kInfo:
		case HighsLogType::kDetailed:
		case HighsLogType::kVerbose:
			print(message);
			break;

		case HighsLogType::kWarning:
			warning(message);
			break;

		case HighsLogType::kError:
			highsLogErrStack.push(message);
			break;
		}
	}

	void throwIfHighsError() {
		if (highsLogErrStack.empty()) return; // No error occured
		std::string msg;
		while (!highsLogErrStack.empty()) {
			msg += highsLogErrStack.top();
			highsLogErrStack.pop();
		}
		throw std::runtime_error(msg);
	}

	MexCallSyntax checkMexCallSyntax(ArgumentList& inputs) {
		if (inputs.size() == 1) {
			if (!(isMatlabString(inputs[0]) && isScalar(inputs[0]))) {
				throw std::runtime_error("Input argument must be a MATLAB string.");
			}
			const TypedArray<MATLABString> in0(inputs[0]);
			const std::string instr = matlabStringToStdString(in0[0]);
			if (instr == "ver") {
				return MexCallSyntax::kVer;
			}
			else if (instr == "defopts") {
				return MexCallSyntax::kDefaultOpts;
			}
			else if (instr == "intType") {
				return MexCallSyntax::kIntType;
			}
			else {
				throw std::runtime_error("Input string is invalid.");
			}
		}
		else if (inputs.size() >= 2 && inputs.size() <= 7) {
			return MexCallSyntax::kIis;
		}
		else {
			throw std::runtime_error("Invalid number of input arguments.");
		}
	}

	void checkHighsReturnStatus(const HighsStatus status, const std::string& warnMsg, const std::string& errMsg) {
		// Throw an error if HiGHS passed the error message via the logging callback
		throwIfHighsError();
		// Check for the return status
		switch (status) {
		case HighsStatus::kError:
			throw std::runtime_error(errMsg);
			break;

		case HighsStatus::kWarning:
			warning(warnMsg);
			break;

		default:
			; // Do nothing
		}
	}

	/* Determine the number of columns implied by the second input argument (A).
	 * Returns false if A is omitted (empty), in which case numCol is not set.
	 */
	bool numColFromMatrixArgIn(ArgumentList& inputs, HighsInt& numCol) {
		if (isEmpty(inputs[1])) return false;
		if (isDouble(inputs[1])) {
			auto const dims = inputs[1].getDimensions();
			if (!isMatrix(dims)) {
				throw std::runtime_error("Second input argument (A) must be a matrix of double type.");
			}
			numCol = castToHighsInt(dims[1]);
			return true;
		}
		if (isCell(inputs[1])) {
			if (numel(inputs[1]) != 5) throw std::runtime_error("Second input argument (A) must be a cell array of 5 elements.");
			const CellArray cell(inputs[1]);
			if (!(isDouble(cell[4]) && isScalar(cell[4]))) {
				throw std::runtime_error("The 5th element of the cell array passed as the second input argument (A) must be a double scalar representing the number of columns of A.");
			}
			const TypedArray<double> ncolsA = cell[4];
			numCol = castToHighsInt(ncolsA[0]);
			return true;
		}
		throw std::runtime_error("Second input argument (A) must be a matrix of double type or, a cell array of 5 elements.");
	}

	/* Process the first input argument (c) and set the number of columns of the LP.
	 * Unlike callhighs, c may be empty, in which case the number of columns is taken
	 * from A and all the column costs are set to zero. Column costs play no role in
	 * the IIS calculation, they are accepted only so that the leading input arguments
	 * match those of callhighs.
	 */
	void process1stArgIn(ArgumentList& inputs, HighsLp& lp) {
		HighsInt numColA = 0;
		const bool haveA = numColFromMatrixArgIn(inputs, numColA);

		if (isEmpty(inputs[0])) {
			if (!haveA) {
				throw std::runtime_error("First input argument (c) may only be empty when the second input argument (A) is not empty.");
			}
			lp.num_col_ = numColA;
			lp.col_cost_.assign(lp.num_col_, 0);
		}
		else {
			auto const dims = inputs[0].getDimensions();
			if (!(isDouble(inputs[0]) && isVector(dims))) {
				throw std::runtime_error("First input argument (c) must be a double type vector, or an empty array.");
			}
			const TypedArray<double> c(inputs[0]);
			lp.num_col_ = castToHighsInt(numel(c));
			lp.col_cost_ = matlabVectorToStdVector(c);
			if (haveA && numColA != lp.num_col_) {
				throw std::runtime_error(formatMessage("The second input argument (A) must have ", lp.num_col_, " columns to match the length of the first input argument (c)."));
			}
		}
		lp.offset_ = 0;

		if constexpr (MexDebugPrinting) {
			print("lp.col_cost_ = "); disp(lp.col_cost_);
		}
	}

	// Pre-condition: lp.num_col_ must be set.
	void process2ndArgIn(ArgumentList& inputs, HighsLp& lp) {
		if (isEmpty(inputs[1])) { // No linear constraints, hence no matrix A
			lp.num_row_ = 0;
			lp.a_matrix_.format_ = MatrixFormat::kColwise;
			lp.a_matrix_.start_.assign(lp.num_col_ + 1, 0);
			lp.a_matrix_.num_col_ = lp.num_col_;
			lp.a_matrix_.num_row_ = 0;
			return;
		}

		bool isSparse = false;
		if (isDouble(inputs[1])) {
			auto const dims = inputs[1].getDimensions();
			// The column count was already checked against c in process1stArgIn
			lp.num_row_ = castToHighsInt(dims[0]);
		}
		else { // Cell array. Already validated in numColFromMatrixArgIn.
			isSparse = true;
			const CellArray cell(inputs[1]);
			if (!(isDouble(cell[3]) && isScalar(cell[3]))) {
				throw std::runtime_error("The 4th element of the cell array passed as the second input argument (A) must be a double scalar representing the number of rows of A.");
			}
			const TypedArray<double> nrowsA = cell[3];
			lp.num_row_ = castToHighsInt(nrowsA[0]);
		}

		lp.a_matrix_.format_ = MatrixFormat::kColwise;
		if (isSparse) {
			const CellArray cell(inputs[1]);
			auto isDoubleVec = [](const Array& arr_) -> bool { return isDouble(arr_) && (isEmpty(arr_) || isVectorArr(arr_)); };
			if (!(isDoubleVec(cell[0]) && isDoubleVec(cell[1]) && isDoubleVec(cell[2]))) {
				throw std::runtime_error("The 1st, 2nd, and 3rd elements of the cell array passed as the second input argument (A) must be double type vectors.");
			}
			const TypedArray<double> iA = cell[0];
			const TypedArray<double> jA = cell[1];
			const TypedArray<double> vA = cell[2];
			auto const nA = numel(iA);
			if (!(nA == numel(jA) && nA == numel(vA))) {
				throw std::runtime_error("The 1st, 2nd, and 3rd elements of the cell array passed as the second input argument (A) must be vectors of the same length.");
			}
			matlabMatrixToHighsFormat(lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_,
				iA, jA, vA, lp.num_row_, lp.num_col_);
		}
		else {
			const TypedArray<double> A(inputs[1]);
			matlabMatrixToHighsFormat(lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_,
				A, lp.num_row_, lp.num_col_);
		}
		lp.a_matrix_.num_col_ = lp.num_col_;
		lp.a_matrix_.num_row_ = lp.num_row_;

		if constexpr (MexDebugPrinting) {
			print("lp.a_matrix_.start_ = "); disp(lp.a_matrix_.start_);
			print("lp.a_matrix_.index_ = "); disp(lp.a_matrix_.index_);
			print("lp.a_matrix_.value_ = "); disp(lp.a_matrix_.value_);
		}
	}

	/* Process one of the four bound vectors (L, U, l, u).
	 * Pre-condition: numExpected is the number of rows (for L and U) or the number
	 * of columns (for l and u) of the LP.
	 */
	void processBoundArgIn(ArgumentList& inputs, std::vector<double>& bound, const size_t argIndx,
		const HighsInt numExpected, const double defaultValue, const std::string& argDescription) {
		if (!numExpected) { // Nothing to bound, e.g. L and U when there are no linear constraints
			if (argIndx < inputs.size() && !isEmpty(inputs[argIndx])) {
				throw std::runtime_error(formatMessage("Expected ", argDescription, " to be an empty array."));
			}
			return;
		}

		bool setToDefault = true;
		if (argIndx < inputs.size() && !isEmpty(inputs[argIndx])) {
			auto const dims = inputs[argIndx].getDimensions();
			if (!(isDouble(inputs[argIndx]) && isVector(dims))) {
				throw std::runtime_error(formatMessage(argDescription, " must be a vector of double type, or an empty array."));
			}
			setToDefault = false;
		}

		if (setToDefault) {
			bound.assign(numExpected, defaultValue);
		}
		else {
			const TypedArray<double> b(inputs[argIndx]);
			if (castToHighsInt(numel(b)) != numExpected) {
				throw std::runtime_error(formatMessage("Expected length of ", argDescription, " to be ", numExpected, "."));
			}
			bound = matlabVectorToStdVector(b);
		}

		if constexpr (MexDebugPrinting) {
			print(formatMessage(argDescription, " = ")); disp(bound);
		}
	}

	// Process the seventh input argument (options).
	void processOptionsArgIn(ArgumentList& inputs, Highs& highs) {
		if (inputs.size() <= 6 || isEmpty(inputs[6])) return; // Use the default options
		if (!(isStruct(inputs[6]) && isScalar(inputs[6]))) {
			throw std::runtime_error("Seventh input argument (options) must be a 1x1 MATLAB struct, or an empty array.");
		}
		const StructArray opts(inputs[6]);
		setHighsOptions(highs, opts, "seventh");
	}

	void runIis(ArgumentList& inputs, Highs& highs, HighsIis& iis) {
		// Build the LP from the input arguments
		HighsLp lp;
		process1stArgIn(inputs, lp);
		process2ndArgIn(inputs, lp);
		processBoundArgIn(inputs, lp.row_lower_, 2, lp.num_row_, -kHighsInf, "third input argument (L)");
		processBoundArgIn(inputs, lp.row_upper_, 3, lp.num_row_, kHighsInf, "fourth input argument (U)");
		processBoundArgIn(inputs, lp.col_lower_, 4, lp.num_col_, -kHighsInf, "fifth input argument (l)");
		processBoundArgIn(inputs, lp.col_upper_, 5, lp.num_col_, kHighsInf, "sixth input argument (u)");

		/* Ask for a full IIS calculation by default. This is done before applying the
		 * user supplied options so that the user can override it.
		 */
		if (highs.setOptionValue("iis_strategy", kHighsMexDefaultIisStrategy) != HighsStatus::kOk) {
			throw std::runtime_error("Failed to set the HiGHS option \"iis_strategy\".");
		}

		// Set the user supplied options
		processOptionsArgIn(inputs, highs);

		bool log_to_console;
		highs.getOptionValue("log_to_console", log_to_console);
		if (log_to_console) {
			// Set callback with Highs
			auto callback = [this](int callbackType, const std::string& message, const HighsCallbackOutput* dataOut, HighsCallbackInput*, void*) -> void {
				logCallback(callbackType, static_cast<HighsLogType>(dataOut->log_type), message);
				};
			if (highs.setCallback(callback, nullptr) != HighsStatus::kOk) {
				throw std::runtime_error("Failed to set the logging callback with HiGHS.");
			}
			checkHighsReturnStatus(highs.startCallback(HighsCallbackType::kCallbackLogging),
				"Warning issued when attempting to start the logging callback.",
				"Failed to start the logging callback.");
			highs.setOptionValue("log_to_console", false);
		}

		// Pass the LP to HiGHS
		checkHighsReturnStatus(highs.passModel(lp),
			"Warning issued when passing the model to the HiGHS solver.",
			"Failed to pass the model to the HiGHS solver.");

		/* Compute the IIS. Highs::getIis solves the model first if its status is not
		 * yet known, so there is no need to call Highs::run beforehand.
		 */
		checkHighsReturnStatus(highs.getIis(iis),
			"Warning issued when computing the irreducible infeasible subsystem (IIS). The returned subsystem may not be valid, check the \"valid\" field of the first output argument.",
			"Failed to compute the irreducible infeasible subsystem (IIS).");
	}

public:

	/* This is the gateway routine for the MEX-file. */
	void operator()(ArgumentList outputs, ArgumentList inputs) {
		try {
			Highs highs;
			HighsIis iis;

			switch (checkMexCallSyntax(inputs)) {
			case MexCallSyntax::kVer:
				if (outputs.size() != 1) throw std::runtime_error("Number of output arguments must be one.");
				outputs[0] = factory.createScalar(highs.version());
				return;

			case MexCallSyntax::kDefaultOpts:
				if (outputs.size() != 1) throw std::runtime_error("Number of output arguments must be one.");
				outputs[0] = highsOptionsToMatlabStruct(highs, true);
				return;

			case MexCallSyntax::kIntType:
				if (outputs.size() != 1) throw std::runtime_error("Number of output arguments must be one.");
				outputs[0] = factory.createScalar(HighsInt2MatlabClassStr);
				return;

			case MexCallSyntax::kIis:
				if (!(outputs.size() >= 1 && outputs.size() <= 4)) {
					throw std::runtime_error("Number of output arguments must be >= 1 and <= 4.");
				}
				// Clear the error stack
				while (!highsLogErrStack.empty()) {
					highsLogErrStack.pop();
				}
				// Process inputs and compute the IIS
				runIis(inputs, highs, iis);
				// Assign outputs
				outputs[0] = highsIisToMatlabStruct(iis);
				if (outputs.size() > 1) {
					outputs[1] = highsIisLpToMatlabStruct(iis);
					if (outputs.size() > 2) {
						outputs[2] = highsInfoToMatlabStruct(highs);
						if (outputs.size() > 3) {
							outputs[3] = highsOptionsToMatlabStruct(highs, false);
						}
					}
				}
				return;
			}
		}
		catch (const matlab::engine::Exception& excpt) {
			error__(excpt.what());
		}
		catch (const matlab::Exception& excpt) {
			error__(excpt.what());
		}
		catch (const std::exception& excpt) {
			error__(excpt.what());
		}
		catch (...) {
			error__("Unexpected error.");
		}
	}

};

// EOF
