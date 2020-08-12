#ifndef CVM_C_API_GRAPH_H_
#define CVM_C_API_GRAPH_H_

/*! \brief CVM_DLL prefix for windows */
#ifdef _WIN32
#ifdef CVM_EXPORTS
#define CVM_DLL __declspec(dllexport)
#else
#define CVM_DLL __declspec(dllimport)
#endif
#else
#define CVM_DLL __attribute__((visibility("default")))
#endif

/*! \brief manually define unsigned int */
typedef unsigned int nn_uint;

/*! \brief handle to a function that takes param and creates symbol */
typedef void *OpHandle;
/*! \brief handle to a symbol that can be bind as operator */
typedef void *SymbolHandle;
/*! \brief handle to Graph */
typedef void *GraphHandle;

#ifdef __cplusplus
extern "C" {
#endif
// Graph IR API
/*!
 * \brief create a graph handle from symbol
 * \param symbol The symbol representing the graph.
 * \param graph The graph handle created.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMGraphCreate(SymbolHandle symbol, GraphHandle *graph);
/*!
 * \brief free the graph handle
 * \param handle The handle to be freed.
 */
CVM_DLL int CVMGraphFree(GraphHandle handle);
/*!
 * \brief Get a new symbol from the graph.
 * \param graph The graph handle.
 * \param symbol The corresponding symbol
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMGraphGetSymbol(GraphHandle graph, SymbolHandle *symbol);

/*!
 * \brief Get Set a attribute in json format.
 * This feature allows pass graph attributes back and forth in reasonable speed.
 *
 * \param handle The graph handle.
 * \param key The key to the attribute.
 * \param json_value The value need to be in format [type_name, value],
 *  Where type_name is a registered type string in C++ side via DMLC_JSON_ENABLE_ANY.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMGraphSetJSONAttr(GraphHandle handle,
                                const char* key,
                                const char* json_value);

/*!
 * \brief Get a serialized attrirbute from graph.
 * This feature allows pass graph attributes back and forth in reasonable speed.
 *
 * \param handle The graph handle.
 * \param key The key to the attribute.
 * \param json_out The result attribute, can be NULL if the attribute do not exist.
 *  The json_out is an array of [type_name, value].
 *  Where the type_name is a registered type string in C++ side via DMLC_JSON_ENABLE_ANY.
 * \param success Whether the result is contained in out.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMGraphGetJSONAttr(GraphHandle handle,
                                const char* key,
                                const char** json_out,
                                int *success);

/*!
 * \brief Set a attribute whose type is std::vector<NodeEntry> in c++
 * This feature allows pass List of symbolic variables for gradient request.
 *
 * \note This is beta feature only used for test purpos
 *
 * \param handle The graph handle.
 * \param key The key to the attribute.
 * \param list The symbol whose outputs represents the list of NodeEntry to be passed.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMGraphSetNodeEntryListAttr_(GraphHandle handle,
                                          const char* key,
                                          SymbolHandle list);
/*!
 * \brief Apply passes on the src graph.
 * \param src The source graph handle.
 * \param num_pass The number of pass to be applied.
 * \param pass_names The names of the pass.
 * \param dst The result graph.
 * \return 0 when success, -1 when failure happens
 */
CVM_DLL int CVMGraphApplyPasses(GraphHandle src,
                                nn_uint num_pass,
                                const char** pass_names,
                                GraphHandle *dst);

#ifdef __cplusplus
} /* end extern "C" */
#endif

#endif  // CVM_C_API_H_
