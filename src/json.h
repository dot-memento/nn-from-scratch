/**
 * @file json.h
 * @brief Header for a lightweight JSON parser and manipulation library.
 *
 * Simple library for parsing, accessing, and modifying JSON data.
 * Supports all JSON types and provides a clean API for working with JSON structures.
 *
 * @author Michael Teixeira
 * @copyright MIT License
 * @see https://github.com/dot-memento/json-lib
 */

#ifndef JSON_H
#define JSON_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdio.h>
#include <stddef.h>
#include <stdbool.h>

/**
 * @enum json_type
 * @brief Represents the type of a JSON value.
 */
typedef enum json_type {
    JSON_NULL,    /**< Null value */
    JSON_BOOL,    /**< Boolean value */
    JSON_NUMBER,  /**< Number value */
    JSON_STRING,  /**< String value */
    JSON_ARRAY,   /**< Array value */
    JSON_OBJECT   /**< Object value */
} json_type;

/**
 * @enum json_error
 * @brief Represents error codes for JSON operations.
 */
typedef enum json_error {
    JSON_SUCCESS,                  /**< Operation successful */
    JSON_ERROR_ALLOCATION,         /**< Allocation error */
    JSON_ERROR_NULL,               /**< Null pointer error */
    JSON_ERROR_WRONG_TYPE,         /**< Type mismatch error */
    JSON_ERROR_INDEX_OUT_OF_BOUNDS,/**< Array index out of bounds */
    JSON_ERROR_KEY_NOT_FOUND,      /**< Object key not found */
    JSON_ERROR_IO,                 /**< I/O error */
    JSON_ERROR_INVALID_OPTIONS,    /**< Invalid option(s) */
    JSON_ERROR_MAX_DEPTH,          /**< Maximum depth exceeded */
    JSON_ERROR_NUMBER_FORMAT,      /**< Number format error */
    JSON_ERROR_ESCAPE_SEQUENCE,    /**< Invalid escape sequence */
    JSON_ERROR_UNICODE,            /**< Unicode error */
    JSON_ERROR_BUFFER_TOO_SMALL,   /**< Buffer too small */
    JSON_ERROR_CIRCULAR_REFERENCE, /**< Circular reference error */
    JSON_ERROR_UNEXPECTED_CHARACTER, /**< Unexpected character error */
    JSON_ERROR_UNEXPECTED_IDENTIFIER  /**< Unexpected identifier error */
} json_error;

/**
 * @struct json_value
 * @brief Represents a JSON value.
 */
typedef struct json_value json_value;

/**
 * @struct json_error_info
 * @brief Detailed information about JSON errors.
 */
typedef struct json_error_info {
    json_error error;        /**< Error code */
    unsigned int line;       /**< Line number at error */
    unsigned int column;     /**< Column number at error */
    char message[256];       /**< Error message */
} json_error_info;

/**
 * @struct json_parse_options
 * @brief Options for parsing JSON.
 */
typedef struct json_parse_options {
    json_error_info *error_info; /**< Optional pointer to error info for detailed errors (not allocated by the parser, has to be provided or NULL) */
    size_t max_depth;            /**< Maximum allowed nesting depth (default is 1000) */
} json_parse_options;

/**
 * @struct json_format_options
 * @brief Options for formatting (serializing) JSON output.
 */
typedef struct json_format_options {
    json_error_info *error_info; /**< Optional pointer to error info for detailed errors */
    size_t indent_size;          /**< Number of spaces for indentation (compact if 0, default is 2) */
    size_t max_depth;            /**< Maximum allowed nesting depth (default is 1000) */
} json_format_options;

/**
 * @brief Converts a JSON error code to a string.
 * @param code JSON error code.
 * @return C-string representation of the error.
 */
const char* json_error_to_string(json_error code);

/**
 * @brief Creates a JSON null value.
 * @param[out] out Pointer to store the created JSON value.
 * @return json_error Status code.
 */
json_error json_null_create(json_value **out);

/**
 * @brief Creates a JSON boolean value.
 * @param value Boolean value.
 * @param[out] out Pointer to store the created JSON value.
 * @return json_error Status code.
 */
json_error json_bool_create(bool value, json_value **out);

/**
 * @brief Creates a JSON number value.
 * @param value Number value.
 * @param[out] out Pointer to store the created JSON value.
 * @return json_error Status code.
 */
json_error json_number_create(double value, json_value **out);

/**
 * @brief Creates a JSON string value by copying the provided string.
 * @param value C-string value.
 * @param[out] out Pointer to store the created JSON value.
 * @return json_error Status code.
 */
json_error json_string_create(const char *value, json_value **out);

/**
 * @brief Creates a JSON string value without copying the provided string.
 * @param value C-string value, which becomes owned by the json_entry.
 * @param[out] out Pointer to store the created JSON value.
 * @return json_error Status code.
 */
json_error json_string_create_nocopy(char *value, json_value **out);

/**
 * @brief Creates a JSON array.
 * @param[out] out Pointer to store the created JSON array value.
 * @return json_error Status code.
 */
json_error json_array_create(json_value **out);

/**
 * @brief Creates a JSON object.
 * @param[out] out Pointer to store the created JSON object value.
 * @return json_error Status code.
 */
json_error json_object_create(json_value **out);

/**
 * @brief Creates a deep copy of a JSON value.
 * @param entry JSON value to clone.
 * @param[out] out Pointer to store the cloned JSON value.
 * @return json_error Status code.
 */
json_error json_clone(const json_value *value, json_value **out);

/**
 * @brief Frees the memory allocated for a JSON value.
 * @param entry JSON value to free.
 */
void json_free(json_value *value);

/**
 * @brief Returns the type of a JSON value.
 * @param entry JSON value.
 * @param[out] out Pointer to store the JSON type.
 * @return json_error Status code.
 */
json_error json_get_type(const json_value *value, json_type *out);

/**
 * @brief Gets the boolean value from a JSON boolean entry.
 * @param entry JSON value.
 * @param[out] out Pointer to store the boolean.
 * @return json_error Status code.
 */
json_error json_bool_get(const json_value *value, bool *out);

/**
 * @brief Gets the numeric value from a JSON number entry.
 * @param entry JSON value.
 * @param[out] out Pointer to store the number.
 * @return json_error Status code.
 */
json_error json_number_get(const json_value *value, double *out);

/**
 * @brief Gets the string from a JSON string entry.
 * @param entry JSON value.
 * @param[out] out Pointer to store the C-string.
 * @return json_error Status code.
 */
json_error json_string_get(const json_value *value, const char **out);

/**
 * @brief Changes a JSON value to null.
 * @param entry JSON value to change.
 * @return json_error Status code.
 */
json_error json_set_as_null(json_value *value);

/**
 * @brief Changes a JSON value to a boolean.
 * @param entry JSON value to change.
 * @param value New boolean value.
 * @return json_error Status code.
 */
json_error json_set_as_bool(json_value *value, bool new_value);

/**
 * @brief Changes a JSON value to a number.
 * @param entry JSON value to change.
 * @param value New numeric value.
 * @return json_error Status code.
 */
json_error json_set_as_number(json_value *value, double new_value);

/**
 * @brief Changes a JSON value to a string (by copying).
 * @param entry JSON value to change.
 * @param string New C-string.
 * @return json_error Status code.
 */
json_error json_set_as_string(json_value *value, const char *new_value);

/**
 * @brief Changes a JSON value to a string without copying.
 * @param entry JSON value to change.
 * @param string New C-string, which becomes owned by the json_entry.
 * @return json_error Status code.
 */
json_error json_set_as_string_nocopy(json_value *value, char *new_value);

/**
 * @brief Changes a JSON value to an array.
 * @param entry JSON value to change.
 * @return json_error Status code.
 */
json_error json_set_as_array(json_value *value);

/**
 * @brief Changes a JSON value to an object.
 * @param entry JSON value to change.
 * @return json_error Status code.
 */
json_error json_set_as_object(json_value *value);

/**
 * @brief Returns the length of a JSON array.
 * @param array JSON array value.
 * @param[out] out Pointer to store the length.
 * @return json_error Status code.
 */
json_error json_array_length(const json_value *array, size_t *out);

/**
 * @brief Gets an element from a JSON array.
 * @param array JSON array value.
 * @param index Zero-based index of the element.
 * @param[out] out Pointer to store the JSON element.
 * @return json_error Status code.
 */
json_error json_array_get(const json_value *array, size_t index, json_value **out);

/**
 * @brief Sets an element in a JSON array.
 * @param array JSON array value.
 * @param index Zero-based index of the element.
 * @param value New JSON value.
 * @return json_error Status code.
 */
json_error json_array_set(json_value *array, size_t index, json_value *value);

/**
 * @brief Appends a new element to a JSON array.
 * @param array JSON array value.
 * @param value New JSON value to append.
 * @return json_error Status code.
 */
json_error json_array_append(json_value *array, json_value *value);

/**
 * @brief Inserts an element at a specific position in a JSON array.
 * @param array JSON array value.
 * @param index Index at which to insert the element.
 * @param value New JSON value.
 * @return json_error Status code.
 */
json_error json_array_insert(json_value *array, size_t index, json_value *value);

/**
 * @brief Removes an element from a JSON array.
 * @param array JSON array value.
 * @param index Zero-based index of the element to remove.
 * @param[out] out Optional pointer to store the removed element,
 *                 which becomes owned by the caller. If NULL, the element is freed.
 * @return json_error Status code.
 */
json_error json_array_remove(json_value *array, size_t index, json_value **out);

/**
 * @brief Returns the number of key-value pairs in a JSON object.
 * @param object JSON object value.
 * @param[out] out Pointer to store the size.
 * @return json_error Status code.
 */
json_error json_object_size(const json_value *object, size_t *out);

/**
 * @brief Checks if a JSON object contains a specific key.
 * @param object JSON object value.
 * @param key Key to search for.
 * @param[out] out Pointer to store the result (true if exists).
 * @return json_error Status code.
 */
json_error json_object_has_key(const json_value *object, const char *key, bool *out);

/**
 * @brief Gets a value from a JSON object by key.
 * @param object JSON object value.
 * @param key Key to search for.
 * @param[out] out Pointer to store the JSON value.
 * @return json_error Status code.
 */
json_error json_object_get(const json_value *object, const char *key, json_value **out);

/**
 * @brief Sets a key-value pair in a JSON object.
 * @param object JSON object value.
 * @param key Key for the entry.
 * @param value JSON value to associate with the key.
 * @return json_error Status code.
 */
json_error json_object_set(json_value *object, const char *key, json_value *value);

/**
 * @brief Removes a key-value pair from a JSON object.
 * @param object JSON object value.
 * @param key Key to remove.
 * @param[out] out Optional pointer to store the removed value,
 *                 which becomes owned by the caller. If NULL, the element is freed.
 * @return json_error Status code.
 */
json_error json_object_remove(json_value *object, const char *key, json_value **out);

/**
 * @brief Parses a JSON string.
 * @param string C-string containing the JSON input.
 * @param[out] value Pointer to store the parsed JSON value.
 * @param options Optional parsing options (NULL for default values).
 * @return json_error Status code.
 */
json_error json_parse_string(const char *string, json_value **value, const json_parse_options *options);

/**
 * @brief Parses JSON input from a file.
 * @param file File pointer containing the JSON input.
 * @param[out] value Pointer to store the parsed JSON value.
 * @param options Optional parsing options (NULL for default values).
 * @return json_error Status code.
 */
json_error json_parse_file(FILE *file, json_value **value, const json_parse_options *options);

/**
 * @brief Serializes a JSON value to a file.
 * @param entry JSON value to serialize.
 * @param file File pointer to write the JSON output.
 * @param options Optional formatting options (NULL for default values).
 * @return json_error Status code.
 */
json_error json_serialize_to_file(const json_value *value, FILE *file, const json_format_options *options);

/**
 * @brief Serializes a JSON value to a string.
 * @param entry JSON value to serialize.
 * @param[out] dst Pointer to store the dynamically allocated JSON string.
 * @param options Optional formatting options (NULL for default values).
 * @return json_error Status code.
 */
json_error json_serialize_to_string(const json_value *value, char **dst, const json_format_options *options);

#ifdef __cplusplus
}
#endif

#endif // JSON_H