#ifndef JSON_H
#define JSON_H

#include <stdbool.h>
#include <stdio.h>


/**
 * @brief Represents a JSON entry (value).
 */
typedef struct json_entry json_entry;

/**
 * @brief Enumerates possible JSON entry types.
 */
typedef enum json_type {
    JSON_NULL = 0, /** Represents a JSON null value. */
    JSON_NUMBER,   /** Represents a JSON number (implemented as double). */
    JSON_STRING,   /** Represents a JSON string. */
    JSON_BOOL,     /** Represents a JSON boolean. */
    JSON_ARRAY,    /** Represents a JSON array. */
    JSON_OBJECT    /** Represents a JSON object. */
} json_type;


/**
 * @brief Parses a JSON file and returns the root entry.
 * @param filename The path to the JSON file.
 * @return A pointer to the root \c json_entry, or \c NULL if parsing fails.
 * @note The returned \c json_entry must be freed using \c json_free().
 */
json_entry* json_parse_file(const char *filename);

/**
 * @brief Frees a JSON entry and all its children.
 * @param entry The \c json_entry to free.
 */
void json_free(json_entry *entry);


/**
 * @brief Retrieves a value from a JSON object by key.
 * @param object The JSON object entry.
 * @param key The key to look for.
 * @return A pointer to the \c json_entry associated with the key, or \c NULL if not found or not an object.
 * @note The returned entry is owned by the object and should not be freed separately.
 */
json_entry* json_object_get(const json_entry *object, const char *key);

/**
 * @brief Retrieves a value from a JSON array by index.
 * @param array The JSON array entry.
 * @param index The zero-based index.
 * @return A pointer to the \c json_entry at the given index, or \c NULL if out of bounds or not an array.
 * @note The returned entry is owned by the object and should not be freed separately.
 */
json_entry* json_array_get(const json_entry *array, size_t index);

/**
 * @brief Counts the number of entry in a JSON array.
 * @param array The JSON array entry.
 * @return The number of entry in the array, or \c 0 if not an array.
 */
size_t json_array_count(const json_entry *array);


/**
 * @brief Retrieves a number from a JSON entry.
 * @param entry The JSON entry.
 * @return The numeric value of the entry.
 * @note The behavior is undefined if the entry is not a number.
 */
double json_as_number(const json_entry *entry);

/**
 * @brief Tries to retrieve a number from a JSON entry.
 * @param entry The JSON entry.
 * @param[out] value Output parameter that receives the numerical value,
 *              or unchanged if the entry is not a number.
 * @return \c true if the entry is a number, \c false otherwise.
 */
bool json_try_as_number(const json_entry *entry, double *value);


/**
 * @brief Retrieves a string from a JSON entry.
 * @param entry The JSON entry.
 * @return A pointer to the string of the entry.
 * @note The behavior is undefined if the entry is not a string.
 * @note The returned string is owned by the entry and should not be freed separately.
 */
const char* json_as_string(const json_entry *entry);

/**
 * @brief Tries to retrieve a string from a JSON entry.
 * @param entry The JSON entry.
 * @param[out] string Output parameter that receives the string,
 *               or unchanged if the entry is not a string.
 * @return \c true if the entry is a string, \c false otherwise.
 * @note The returned string is owned by the entry and should not be freed separately.
 */
bool json_try_as_string(const json_entry *entry, const char **string);


/**
 * @brief Retrieves a boolean value from a JSON entry.
 * @param entry The JSON entry.
 * @return The boolean value of the entry.
 * @note The behavior is undefined if the entry is not a boolean.
 */
bool json_as_bool(const json_entry *entry);

/**
 * @brief Tries to retrieve a boolean value from a JSON entry.
 * @param entry The JSON entry.
 * @param[out] value Output parameter that receives the boolean value,
 *              or unchanged if the entry is not a boolean.
 * @return \c true if the entry is a boolean, \c false otherwise.
 */
bool json_try_as_bool(const json_entry *entry, bool *value);


/**
 * @brief Retrieves the type of a JSON entry.
 * @param entry The JSON entry.
 * @return The json_type of the entry.
 */
json_type json_get_type(const json_entry *entry);


/**
 * @brief Creates a new, empty JSON entry.
 * @return A pointer to a newly allocated \c json_entry, or \c NULL if memory allocation fails.
 * @note The caller is responsible for freeing the entry using \c json_free()
 *       or by freeing a parent entry.
 */
json_entry* json_new_entry();


/**
 * @brief Sets a key-value pair in a JSON object.
 * @param object The JSON object entry.
 * @param key The key to set.
 * @param value The \c json_entry to associate with the key.
 * @return \c true on success, \c false on failure
 *         (e.g. if the object is not a JSON object or if memory allocation fails).
 * @note If the key already exists, the old value is freed and replaced with the new one.
 */
bool json_object_set(json_entry *object, const char *key, json_entry *value);

/**
 * @brief Removes a key-value pair from a JSON object and retrieve the removed value.
 * @param object The JSON object entry.
 * @param key The key to remove.
 * @return The removed \c json_entry, or \c NULL if the key was
 *         not found or the entry was not an object.
 * @note The caller becomes responsible for freeing the returned \c json_entry.
 */
json_entry* json_object_remove(json_entry *object, const char *key);


/**
 * @brief Appends an entry to a JSON array.
 * @param array The JSON array entry.
 * @param value The \c json_entry to append.
 * @return \c true on success, \c false on failure
 *         (e.g., if the entry is not a JSON array or if memory allocation fails).
 */
bool json_array_append(json_entry *array, json_entry *value);

/**
 * @brief Inserts an entry into a JSON array at a specified index.
 * @param array The JSON array entry.
 * @param index The zero-based index to insert at.
 * @param value The \c json_entry to insert.
 * @return \c true on success, \c false on failure
 *         (e.g., if the entry is not a JSON array or if memory allocation fails).
 */
bool json_array_insert(json_entry *array, size_t index, json_entry *value);

/**
 * @brief Removes a value from a JSON array at a specified index.
 * @param array The JSON array entry.
 * @param index The zero-based index to remove.
 * @return The removed \c json_entry, or \c NULL if the index was
 *         out of bounds or the entry was not an array.
 * @note The caller becomes responsible for freeing the returned \c json_entry if necessary.
 */
json_entry* json_array_remove(json_entry *array, size_t index);


/**
 * @brief Sets a JSON entry to null.
 * @param entry The JSON entry.
 * @note If the entry was previously an object, array, or string, its memory is freed.
 */
void json_set_null(json_entry *entry);

/**
 * @brief Sets a JSON entry to a numeric value.
 * @param entry The JSON entry.
 * @param value The numeric value to set.
 * @note If the entry was previously an object, array, or string, its memory is freed.
 */
void json_set_number(json_entry *entry, double value);

/**
 * @brief Sets a JSON entry to a string value.
 * @param entry The JSON entry.
 * @param string The string to set.
 * @return \c true on success, \c false on failure (e.g., if memory allocation fails).
 * @note If the entry was previously an object, array, or string, its memory is freed.
 * @note A copy of the string is stored, so the caller can safely free \p string.
 */
bool json_set_string(json_entry *entry, const char *string);

/**
 * @brief Sets a JSON entry to a boolean value.
 * @param entry The JSON entry.
 * @param value The boolean value to set.
 * @note If the entry was previously an object, array, or string, its memory is freed.
 */
void json_set_bool(json_entry *entry, bool value);

/**
 * @brief Converts a JSON entry into an empty object.
 * @param entry The JSON entry.
 * @note If the entry was previously an object, array, or string, its memory is freed.
 */
void json_set_object(json_entry *entry);

/**
 * @brief Converts a JSON entry into an empty array.
 * @param entry The JSON entry.
 * @note If the entry was previously an object, array, or string, its memory is freed.
 */
void json_set_array(json_entry *entry);


/**
 * @brief Writes a JSON entry to a file in a pretty-printed format.
 * @param file The file to print to.
 * @param entry The JSON entry to print.
 */
void json_fprint(FILE *file, const json_entry *entry);

#endif // JSON_H