// Code generated by smithy-go-codegen DO NOT EDIT.

package route53

import (
	"context"
	"fmt"
	awsmiddleware "github.com/aws/aws-sdk-go-v2/aws/middleware"
	"github.com/aws/aws-sdk-go-v2/service/route53/types"
	"github.com/aws/smithy-go/middleware"
	smithyhttp "github.com/aws/smithy-go/transport/http"
)

// Creates a configuration for DNS query logging. After you create a query logging
// configuration, Amazon Route 53 begins to publish log data to an Amazon
// CloudWatch Logs log group.
//
// DNS query logs contain information about the queries that Route 53 receives for
// a specified public hosted zone, such as the following:
//
//   - Route 53 edge location that responded to the DNS query
//
//   - Domain or subdomain that was requested
//
//   - DNS record type, such as A or AAAA
//
//   - DNS response code, such as NoError or ServFail
//
// Log Group and Resource Policy Before you create a query logging configuration,
// perform the following operations.
//
// If you create a query logging configuration using the Route 53 console, Route
// 53 performs these operations automatically.
//
//   - Create a CloudWatch Logs log group, and make note of the ARN, which you
//     specify when you create a query logging configuration. Note the following:
//
//   - You must create the log group in the us-east-1 region.
//
//   - You must use the same Amazon Web Services account to create the log group
//     and the hosted zone that you want to configure query logging for.
//
//   - When you create log groups for query logging, we recommend that you use a
//     consistent prefix, for example:
//
// /aws/route53/hosted zone name
//
// In the next step, you'll create a resource policy, which controls access to one
//
//	or more log groups and the associated Amazon Web Services resources, such as
//	Route 53 hosted zones. There's a limit on the number of resource policies that
//	you can create, so we recommend that you use a consistent prefix so you can use
//	the same resource policy for all the log groups that you create for query
//	logging.
//
//	- Create a CloudWatch Logs resource policy, and give it the permissions that
//	Route 53 needs to create log streams and to send query logs to log streams. You
//	must create the CloudWatch Logs resource policy in the us-east-1 region. For the
//	value of Resource , specify the ARN for the log group that you created in the
//	previous step. To use the same resource policy for all the CloudWatch Logs log
//	groups that you created for query logging configurations, replace the hosted
//	zone name with * , for example:
//
// arn:aws:logs:us-east-1:123412341234:log-group:/aws/route53/*
//
// To avoid the confused deputy problem, a security issue where an entity without
//
//	a permission for an action can coerce a more-privileged entity to perform it,
//	you can optionally limit the permissions that a service has to a resource in a
//	resource-based policy by supplying the following values:
//
//	- For aws:SourceArn , supply the hosted zone ARN used in creating the query
//	logging configuration. For example, aws:SourceArn:
//	arn:aws:route53:::hostedzone/hosted zone ID .
//
//	- For aws:SourceAccount , supply the account ID for the account that creates
//	the query logging configuration. For example, aws:SourceAccount:111111111111 .
//
// For more information, see [The confused deputy problem]in the Amazon Web Services IAM User Guide.
//
// You can't use the CloudWatch console to create or edit a resource policy. You
//
//	must use the CloudWatch API, one of the Amazon Web Services SDKs, or the CLI.
//
// Log Streams and Edge Locations When Route 53 finishes creating the
// configuration for DNS query logging, it does the following:
//
//   - Creates a log stream for an edge location the first time that the edge
//     location responds to DNS queries for the specified hosted zone. That log stream
//     is used to log all queries that Route 53 responds to for that edge location.
//
//   - Begins to send query logs to the applicable log stream.
//
// The name of each log stream is in the following format:
//
//	hosted zone ID/edge location code
//
// The edge location code is a three-letter code and an arbitrarily assigned
// number, for example, DFW3. The three-letter code typically corresponds with the
// International Air Transport Association airport code for an airport near the
// edge location. (These abbreviations might change in the future.) For a list of
// edge locations, see "The Route 53 Global Network" on the [Route 53 Product Details]page.
//
// Queries That Are Logged Query logs contain only the queries that DNS resolvers
// forward to Route 53. If a DNS resolver has already cached the response to a
// query (such as the IP address for a load balancer for example.com), the resolver
// will continue to return the cached response. It doesn't forward another query to
// Route 53 until the TTL for the corresponding resource record set expires.
// Depending on how many DNS queries are submitted for a resource record set, and
// depending on the TTL for that resource record set, query logs might contain
// information about only one query out of every several thousand queries that are
// submitted to DNS. For more information about how DNS works, see [Routing Internet Traffic to Your Website or Web Application]in the Amazon
// Route 53 Developer Guide.
//
// Log File Format For a list of the values in each query log and the format of
// each value, see [Logging DNS Queries]in the Amazon Route 53 Developer Guide.
//
// Pricing For information about charges for query logs, see [Amazon CloudWatch Pricing].
//
// How to Stop Logging If you want Route 53 to stop sending query logs to
// CloudWatch Logs, delete the query logging configuration. For more information,
// see [DeleteQueryLoggingConfig].
//
// [The confused deputy problem]: https://docs.aws.amazon.com/IAM/latest/UserGuide/confused-deputy.html
// [DeleteQueryLoggingConfig]: https://docs.aws.amazon.com/Route53/latest/APIReference/API_DeleteQueryLoggingConfig.html
// [Routing Internet Traffic to Your Website or Web Application]: https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/welcome-dns-service.html
// [Route 53 Product Details]: http://aws.amazon.com/route53/details/
// [Amazon CloudWatch Pricing]: http://aws.amazon.com/cloudwatch/pricing/
// [Logging DNS Queries]: https://docs.aws.amazon.com/Route53/latest/DeveloperGuide/query-logs.html
func (c *Client) CreateQueryLoggingConfig(ctx context.Context, params *CreateQueryLoggingConfigInput, optFns ...func(*Options)) (*CreateQueryLoggingConfigOutput, error) {
	if params == nil {
		params = &CreateQueryLoggingConfigInput{}
	}

	result, metadata, err := c.invokeOperation(ctx, "CreateQueryLoggingConfig", params, optFns, c.addOperationCreateQueryLoggingConfigMiddlewares)
	if err != nil {
		return nil, err
	}

	out := result.(*CreateQueryLoggingConfigOutput)
	out.ResultMetadata = metadata
	return out, nil
}

type CreateQueryLoggingConfigInput struct {

	// The Amazon Resource Name (ARN) for the log group that you want to Amazon Route
	// 53 to send query logs to. This is the format of the ARN:
	//
	// arn:aws:logs:region:account-id:log-group:log_group_name
	//
	// To get the ARN for a log group, you can use the CloudWatch console, the [DescribeLogGroups] API
	// action, the [describe-log-groups]command, or the applicable command in one of the Amazon Web
	// Services SDKs.
	//
	// [describe-log-groups]: https://docs.aws.amazon.com/cli/latest/reference/logs/describe-log-groups.html
	// [DescribeLogGroups]: https://docs.aws.amazon.com/AmazonCloudWatchLogs/latest/APIReference/API_DescribeLogGroups.html
	//
	// This member is required.
	CloudWatchLogsLogGroupArn *string

	// The ID of the hosted zone that you want to log queries for. You can log queries
	// only for public hosted zones.
	//
	// This member is required.
	HostedZoneId *string

	noSmithyDocumentSerde
}

type CreateQueryLoggingConfigOutput struct {

	// The unique URL representing the new query logging configuration.
	//
	// This member is required.
	Location *string

	// A complex type that contains the ID for a query logging configuration, the ID
	// of the hosted zone that you want to log queries for, and the ARN for the log
	// group that you want Amazon Route 53 to send query logs to.
	//
	// This member is required.
	QueryLoggingConfig *types.QueryLoggingConfig

	// Metadata pertaining to the operation's result.
	ResultMetadata middleware.Metadata

	noSmithyDocumentSerde
}

func (c *Client) addOperationCreateQueryLoggingConfigMiddlewares(stack *middleware.Stack, options Options) (err error) {
	if err := stack.Serialize.Add(&setOperationInputMiddleware{}, middleware.After); err != nil {
		return err
	}
	err = stack.Serialize.Add(&awsRestxml_serializeOpCreateQueryLoggingConfig{}, middleware.After)
	if err != nil {
		return err
	}
	err = stack.Deserialize.Add(&awsRestxml_deserializeOpCreateQueryLoggingConfig{}, middleware.After)
	if err != nil {
		return err
	}
	if err := addProtocolFinalizerMiddlewares(stack, options, "CreateQueryLoggingConfig"); err != nil {
		return fmt.Errorf("add protocol finalizers: %v", err)
	}

	if err = addlegacyEndpointContextSetter(stack, options); err != nil {
		return err
	}
	if err = addSetLoggerMiddleware(stack, options); err != nil {
		return err
	}
	if err = addClientRequestID(stack); err != nil {
		return err
	}
	if err = addComputeContentLength(stack); err != nil {
		return err
	}
	if err = addResolveEndpointMiddleware(stack, options); err != nil {
		return err
	}
	if err = addComputePayloadSHA256(stack); err != nil {
		return err
	}
	if err = addRetry(stack, options); err != nil {
		return err
	}
	if err = addRawResponseToMetadata(stack); err != nil {
		return err
	}
	if err = addRecordResponseTiming(stack); err != nil {
		return err
	}
	if err = addSpanRetryLoop(stack, options); err != nil {
		return err
	}
	if err = addClientUserAgent(stack, options); err != nil {
		return err
	}
	if err = smithyhttp.AddErrorCloseResponseBodyMiddleware(stack); err != nil {
		return err
	}
	if err = smithyhttp.AddCloseResponseBodyMiddleware(stack); err != nil {
		return err
	}
	if err = addSetLegacyContextSigningOptionsMiddleware(stack); err != nil {
		return err
	}
	if err = addTimeOffsetBuild(stack, c); err != nil {
		return err
	}
	if err = addUserAgentRetryMode(stack, options); err != nil {
		return err
	}
	if err = addCredentialSource(stack, options); err != nil {
		return err
	}
	if err = addOpCreateQueryLoggingConfigValidationMiddleware(stack); err != nil {
		return err
	}
	if err = stack.Initialize.Add(newServiceMetadataMiddleware_opCreateQueryLoggingConfig(options.Region), middleware.Before); err != nil {
		return err
	}
	if err = addRecursionDetection(stack); err != nil {
		return err
	}
	if err = addRequestIDRetrieverMiddleware(stack); err != nil {
		return err
	}
	if err = addResponseErrorMiddleware(stack); err != nil {
		return err
	}
	if err = addSanitizeURLMiddleware(stack); err != nil {
		return err
	}
	if err = addRequestResponseLogging(stack, options); err != nil {
		return err
	}
	if err = addDisableHTTPSMiddleware(stack, options); err != nil {
		return err
	}
	if err = addInterceptBeforeRetryLoop(stack, options); err != nil {
		return err
	}
	if err = addInterceptAttempt(stack, options); err != nil {
		return err
	}
	if err = addInterceptExecution(stack, options); err != nil {
		return err
	}
	if err = addInterceptBeforeSerialization(stack, options); err != nil {
		return err
	}
	if err = addInterceptAfterSerialization(stack, options); err != nil {
		return err
	}
	if err = addInterceptBeforeSigning(stack, options); err != nil {
		return err
	}
	if err = addInterceptAfterSigning(stack, options); err != nil {
		return err
	}
	if err = addInterceptTransmit(stack, options); err != nil {
		return err
	}
	if err = addInterceptBeforeDeserialization(stack, options); err != nil {
		return err
	}
	if err = addInterceptAfterDeserialization(stack, options); err != nil {
		return err
	}
	if err = addSpanInitializeStart(stack); err != nil {
		return err
	}
	if err = addSpanInitializeEnd(stack); err != nil {
		return err
	}
	if err = addSpanBuildRequestStart(stack); err != nil {
		return err
	}
	if err = addSpanBuildRequestEnd(stack); err != nil {
		return err
	}
	return nil
}

func newServiceMetadataMiddleware_opCreateQueryLoggingConfig(region string) *awsmiddleware.RegisterServiceMetadata {
	return &awsmiddleware.RegisterServiceMetadata{
		Region:        region,
		ServiceID:     ServiceID,
		OperationName: "CreateQueryLoggingConfig",
	}
}
