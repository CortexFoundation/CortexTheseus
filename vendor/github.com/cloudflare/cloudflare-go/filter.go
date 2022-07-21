package cloudflare

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"

	"errors"
)

var ErrNotEnoughFilterIDsProvided = errors.New("at least one filter ID must be provided.")

// Filter holds the structure of the filter type.
type Filter struct {
	ID          string `json:"id,omitempty"`
	Expression  string `json:"expression"`
	Paused      bool   `json:"paused"`
	Description string `json:"description"`

	// Property is mentioned in documentation however isn't populated in
	// any of the API requests. For now, let's just omit it unless it's
	// provided.
	Ref string `json:"ref,omitempty"`
}

// FiltersDetailResponse is the API response that is returned
// for requesting all filters on a zone.
type FiltersDetailResponse struct {
	Result     []Filter `json:"result"`
	ResultInfo `json:"result_info"`
	Response
}

// FilterDetailResponse is the API response that is returned
// for requesting a single filter on a zone.
type FilterDetailResponse struct {
	Result     Filter `json:"result"`
	ResultInfo `json:"result_info"`
	Response
}

// FilterValidateExpression represents the JSON payload for checking
// an expression.
type FilterValidateExpression struct {
	Expression string `json:"expression"`
}

// FilterValidateExpressionResponse represents the API response for
// checking the expression. It conforms to the JSON API approach however
// we don't need all of the fields exposed.
type FilterValidateExpressionResponse struct {
	Success bool                                `json:"success"`
	Errors  []FilterValidationExpressionMessage `json:"errors"`
}

// FilterValidationExpressionMessage represents the API error message.
type FilterValidationExpressionMessage struct {
	Message string `json:"message"`
}

// Filter returns a single filter in a zone based on the filter ID.
//
// API reference: https://developers.cloudflare.com/firewall/api/cf-filters/get/#get-by-filter-id
func (api *API) Filter(ctx context.Context, zoneID, filterID string) (Filter, error) {
	uri := fmt.Sprintf("/zones/%s/filters/%s", zoneID, filterID)

	res, err := api.makeRequestContext(ctx, http.MethodGet, uri, nil)
	if err != nil {
		return Filter{}, err
	}

	var filterResponse FilterDetailResponse
	err = json.Unmarshal(res, &filterResponse)
	if err != nil {
		return Filter{}, fmt.Errorf("%s: %w", errUnmarshalError, err)
	}

	return filterResponse.Result, nil
}

// Filters returns all filters for a zone.
//
// API reference: https://developers.cloudflare.com/firewall/api/cf-filters/get/#get-all-filters
func (api *API) Filters(ctx context.Context, zoneID string, pageOpts PaginationOptions) ([]Filter, error) {
	uri := buildURI(fmt.Sprintf("/zones/%s/filters", zoneID), pageOpts)

	res, err := api.makeRequestContext(ctx, http.MethodGet, uri, nil)
	if err != nil {
		return []Filter{}, err
	}

	var filtersResponse FiltersDetailResponse
	err = json.Unmarshal(res, &filtersResponse)
	if err != nil {
		return []Filter{}, fmt.Errorf("%s: %w", errUnmarshalError, err)
	}

	return filtersResponse.Result, nil
}

// CreateFilters creates new filters.
//
// API reference: https://developers.cloudflare.com/firewall/api/cf-filters/post/
func (api *API) CreateFilters(ctx context.Context, zoneID string, filters []Filter) ([]Filter, error) {
	uri := fmt.Sprintf("/zones/%s/filters", zoneID)

	res, err := api.makeRequestContext(ctx, http.MethodPost, uri, filters)
	if err != nil {
		return []Filter{}, err
	}

	var filtersResponse FiltersDetailResponse
	err = json.Unmarshal(res, &filtersResponse)
	if err != nil {
		return []Filter{}, fmt.Errorf("%s: %w", errUnmarshalError, err)
	}

	return filtersResponse.Result, nil
}

// UpdateFilter updates a single filter.
//
// API reference: https://developers.cloudflare.com/firewall/api/cf-filters/put/#update-a-single-filter
func (api *API) UpdateFilter(ctx context.Context, zoneID string, filter Filter) (Filter, error) {
	if filter.ID == "" {
		return Filter{}, fmt.Errorf("filter ID cannot be empty")
	}

	uri := fmt.Sprintf("/zones/%s/filters/%s", zoneID, filter.ID)

	res, err := api.makeRequestContext(ctx, http.MethodPut, uri, filter)
	if err != nil {
		return Filter{}, err
	}

	var filterResponse FilterDetailResponse
	err = json.Unmarshal(res, &filterResponse)
	if err != nil {
		return Filter{}, fmt.Errorf("%s: %w", errUnmarshalError, err)
	}

	return filterResponse.Result, nil
}

// UpdateFilters updates many filters at once.
//
// API reference: https://developers.cloudflare.com/firewall/api/cf-filters/put/#update-multiple-filters
func (api *API) UpdateFilters(ctx context.Context, zoneID string, filters []Filter) ([]Filter, error) {
	for _, filter := range filters {
		if filter.ID == "" {
			return []Filter{}, fmt.Errorf("filter ID cannot be empty")
		}
	}

	uri := fmt.Sprintf("/zones/%s/filters", zoneID)

	res, err := api.makeRequestContext(ctx, http.MethodPut, uri, filters)
	if err != nil {
		return []Filter{}, err
	}

	var filtersResponse FiltersDetailResponse
	err = json.Unmarshal(res, &filtersResponse)
	if err != nil {
		return []Filter{}, fmt.Errorf("%s: %w", errUnmarshalError, err)
	}

	return filtersResponse.Result, nil
}

// DeleteFilter deletes a single filter.
//
// API reference: https://developers.cloudflare.com/firewall/api/cf-filters/delete/#delete-a-single-filter
func (api *API) DeleteFilter(ctx context.Context, zoneID, filterID string) error {
	if filterID == "" {
		return fmt.Errorf("filter ID cannot be empty")
	}

	uri := fmt.Sprintf("/zones/%s/filters/%s", zoneID, filterID)

	_, err := api.makeRequestContext(ctx, http.MethodDelete, uri, nil)
	if err != nil {
		return err
	}

	return nil
}

// DeleteFilters deletes multiple filters.
//
// API reference: https://developers.cloudflare.com/firewall/api/cf-filters/delete/#delete-multiple-filters
func (api *API) DeleteFilters(ctx context.Context, zoneID string, filterIDs []string) error {
	if len(filterIDs) == 0 {
		return ErrNotEnoughFilterIDsProvided
	}

	// Swap this to a typed struct and passing in all parameters together.
	q := url.Values{}
	for _, id := range filterIDs {
		q.Add("id", id)
	}

	uri := (&url.URL{
		Path:     fmt.Sprintf("/zones/%s/filters", zoneID),
		RawQuery: q.Encode(),
	}).String()

	_, err := api.makeRequestContext(ctx, http.MethodDelete, uri, nil)
	if err != nil {
		return err
	}

	return nil
}

// ValidateFilterExpression checks correctness of a filter expression.
//
// API reference: https://developers.cloudflare.com/firewall/api/cf-filters/validation/
func (api *API) ValidateFilterExpression(ctx context.Context, expression string) error {
	expressionPayload := FilterValidateExpression{Expression: expression}

	_, err := api.makeRequestContext(ctx, http.MethodPost, "/filters/validate-expr", expressionPayload)
	if err != nil {
		var filterValidationResponse FilterValidateExpressionResponse

		jsonErr := json.Unmarshal([]byte(err.Error()), &filterValidationResponse)
		if jsonErr != nil {
			return fmt.Errorf(errUnmarshalError+": %w", jsonErr)
		}

		if !filterValidationResponse.Success {
			// Unsure why but the API returns `errors` as an array but it only
			// ever shows the issue with one problem at a time ¯\_(ツ)_/¯
			return fmt.Errorf(filterValidationResponse.Errors[0].Message)
		}
	}

	return nil
}
