package congestion

import (
	"time"

	"github.com/lucas-clemente/quic-go/internal/protocol"
	"github.com/lucas-clemente/quic-go/internal/utils"
)

const (
	// maxDatagramSize is the default maximum packet size used in the Linux TCP implementation.
	// Used in QUIC for congestion window computations in bytes.
	maxDatagramSize                 = protocol.ByteCount(protocol.MaxPacketSizeIPv4)
	maxBurstBytes                   = 3 * maxDatagramSize
	renoBeta                float32 = 0.7 // Reno backoff factor.
	maxCongestionWindow             = protocol.MaxCongestionWindowPackets * maxDatagramSize
	minCongestionWindow             = 2 * maxDatagramSize
	initialCongestionWindow         = 32 * maxDatagramSize
)

type cubicSender struct {
	hybridSlowStart HybridSlowStart
	rttStats        *RTTStats
	stats           connectionStats
	cubic           *Cubic

	reno bool

	// Track the largest packet that has been sent.
	largestSentPacketNumber protocol.PacketNumber

	// Track the largest packet that has been acked.
	largestAckedPacketNumber protocol.PacketNumber

	// Track the largest packet number outstanding when a CWND cutback occurs.
	largestSentAtLastCutback protocol.PacketNumber

	// Whether the last loss event caused us to exit slowstart.
	// Used for stats collection of slowstartPacketsLost
	lastCutbackExitedSlowstart bool

	// When true, exit slow start with large cutback of congestion window.
	slowStartLargeReduction bool

	// Congestion window in packets.
	congestionWindow protocol.ByteCount

	// Minimum congestion window in packets.
	minCongestionWindow protocol.ByteCount

	// Maximum congestion window.
	maxCongestionWindow protocol.ByteCount

	// Slow start congestion window in bytes, aka ssthresh.
	slowstartThreshold protocol.ByteCount

	// Number of connections to simulate.
	numConnections int

	// ACK counter for the Reno implementation.
	numAckedPackets uint64

	initialCongestionWindow    protocol.ByteCount
	initialMaxCongestionWindow protocol.ByteCount

	minSlowStartExitWindow protocol.ByteCount
}

var _ SendAlgorithm = &cubicSender{}
var _ SendAlgorithmWithDebugInfos = &cubicSender{}

// NewCubicSender makes a new cubic sender
func NewCubicSender(clock Clock, rttStats *RTTStats, reno bool) *cubicSender {
	return newCubicSender(clock, rttStats, reno, initialCongestionWindow, maxCongestionWindow)
}

func newCubicSender(clock Clock, rttStats *RTTStats, reno bool, initialCongestionWindow, initialMaxCongestionWindow protocol.ByteCount) *cubicSender {
	return &cubicSender{
		rttStats:                   rttStats,
		largestSentPacketNumber:    protocol.InvalidPacketNumber,
		largestAckedPacketNumber:   protocol.InvalidPacketNumber,
		largestSentAtLastCutback:   protocol.InvalidPacketNumber,
		initialCongestionWindow:    initialCongestionWindow,
		initialMaxCongestionWindow: initialMaxCongestionWindow,
		congestionWindow:           initialCongestionWindow,
		minCongestionWindow:        minCongestionWindow,
		slowstartThreshold:         initialMaxCongestionWindow,
		maxCongestionWindow:        initialMaxCongestionWindow,
		numConnections:             defaultNumConnections,
		cubic:                      NewCubic(clock),
		reno:                       reno,
	}
}

// TimeUntilSend returns when the next packet should be sent.
func (c *cubicSender) TimeUntilSend(bytesInFlight protocol.ByteCount) time.Duration {
	return c.rttStats.SmoothedRTT() * time.Duration(maxDatagramSize) / time.Duration(2*c.GetCongestionWindow())
}

func (c *cubicSender) OnPacketSent(
	sentTime time.Time,
	bytesInFlight protocol.ByteCount,
	packetNumber protocol.PacketNumber,
	bytes protocol.ByteCount,
	isRetransmittable bool,
) {
	if !isRetransmittable {
		return
	}
	c.largestSentPacketNumber = packetNumber
	c.hybridSlowStart.OnPacketSent(packetNumber)
}

func (c *cubicSender) CanSend(bytesInFlight protocol.ByteCount) bool {
	return bytesInFlight < c.GetCongestionWindow()
}

func (c *cubicSender) InRecovery() bool {
	return c.largestAckedPacketNumber != protocol.InvalidPacketNumber && c.largestAckedPacketNumber <= c.largestSentAtLastCutback
}

func (c *cubicSender) InSlowStart() bool {
	return c.GetCongestionWindow() < c.GetSlowStartThreshold()
}

func (c *cubicSender) GetCongestionWindow() protocol.ByteCount {
	return c.congestionWindow
}

func (c *cubicSender) GetSlowStartThreshold() protocol.ByteCount {
	return c.slowstartThreshold
}

func (c *cubicSender) ExitSlowstart() {
	c.slowstartThreshold = c.congestionWindow
}

func (c *cubicSender) SlowstartThreshold() protocol.ByteCount {
	return c.slowstartThreshold
}

func (c *cubicSender) MaybeExitSlowStart() {
	if c.InSlowStart() && c.hybridSlowStart.ShouldExitSlowStart(c.rttStats.LatestRTT(), c.rttStats.MinRTT(), c.GetCongestionWindow()/maxDatagramSize) {
		c.ExitSlowstart()
	}
}

func (c *cubicSender) OnPacketAcked(
	ackedPacketNumber protocol.PacketNumber,
	ackedBytes protocol.ByteCount,
	priorInFlight protocol.ByteCount,
	eventTime time.Time,
) {
	c.largestAckedPacketNumber = utils.MaxPacketNumber(ackedPacketNumber, c.largestAckedPacketNumber)
	if c.InRecovery() {
		return
	}
	c.maybeIncreaseCwnd(ackedPacketNumber, ackedBytes, priorInFlight, eventTime)
	if c.InSlowStart() {
		c.hybridSlowStart.OnPacketAcked(ackedPacketNumber)
	}
}

func (c *cubicSender) OnPacketLost(
	packetNumber protocol.PacketNumber,
	lostBytes protocol.ByteCount,
	priorInFlight protocol.ByteCount,
) {
	// TCP NewReno (RFC6582) says that once a loss occurs, any losses in packets
	// already sent should be treated as a single loss event, since it's expected.
	if packetNumber <= c.largestSentAtLastCutback {
		if c.lastCutbackExitedSlowstart {
			c.stats.slowstartPacketsLost++
			c.stats.slowstartBytesLost += lostBytes
			if c.slowStartLargeReduction {
				// Reduce congestion window by lost_bytes for every loss.
				c.congestionWindow = utils.MaxByteCount(c.congestionWindow-lostBytes, c.minSlowStartExitWindow)
				c.slowstartThreshold = c.congestionWindow
			}
		}
		return
	}
	c.lastCutbackExitedSlowstart = c.InSlowStart()
	if c.InSlowStart() {
		c.stats.slowstartPacketsLost++
	}

	// TODO(chromium): Separate out all of slow start into a separate class.
	if c.slowStartLargeReduction && c.InSlowStart() {
		if c.congestionWindow >= 2*c.initialCongestionWindow {
			c.minSlowStartExitWindow = c.congestionWindow / 2
		}
		c.congestionWindow -= maxDatagramSize
	} else if c.reno {
		c.congestionWindow = protocol.ByteCount(float32(c.congestionWindow) * c.RenoBeta())
	} else {
		c.congestionWindow = c.cubic.CongestionWindowAfterPacketLoss(c.congestionWindow)
	}
	if c.congestionWindow < c.minCongestionWindow {
		c.congestionWindow = c.minCongestionWindow
	}
	c.slowstartThreshold = c.congestionWindow
	c.largestSentAtLastCutback = c.largestSentPacketNumber
	// reset packet count from congestion avoidance mode. We start
	// counting again when we're out of recovery.
	c.numAckedPackets = 0
}

func (c *cubicSender) RenoBeta() float32 {
	// kNConnectionBeta is the backoff factor after loss for our N-connection
	// emulation, which emulates the effective backoff of an ensemble of N
	// TCP-Reno connections on a single loss event. The effective multiplier is
	// computed as:
	return (float32(c.numConnections) - 1. + renoBeta) / float32(c.numConnections)
}

// Called when we receive an ack. Normal TCP tracks how many packets one ack
// represents, but quic has a separate ack for each packet.
func (c *cubicSender) maybeIncreaseCwnd(
	_ protocol.PacketNumber,
	ackedBytes protocol.ByteCount,
	priorInFlight protocol.ByteCount,
	eventTime time.Time,
) {
	// Do not increase the congestion window unless the sender is close to using
	// the current window.
	if !c.isCwndLimited(priorInFlight) {
		c.cubic.OnApplicationLimited()
		return
	}
	if c.congestionWindow >= c.maxCongestionWindow {
		return
	}
	if c.InSlowStart() {
		// TCP slow start, exponential growth, increase by one for each ACK.
		c.congestionWindow += maxDatagramSize
		return
	}
	// Congestion avoidance
	if c.reno {
		// Classic Reno congestion avoidance.
		c.numAckedPackets++
		// Divide by num_connections to smoothly increase the CWND at a faster
		// rate than conventional Reno.
		if c.numAckedPackets*uint64(c.numConnections) >= uint64(c.congestionWindow)/uint64(maxDatagramSize) {
			c.congestionWindow += maxDatagramSize
			c.numAckedPackets = 0
		}
	} else {
		c.congestionWindow = utils.MinByteCount(c.maxCongestionWindow, c.cubic.CongestionWindowAfterAck(ackedBytes, c.congestionWindow, c.rttStats.MinRTT(), eventTime))
	}
}

func (c *cubicSender) isCwndLimited(bytesInFlight protocol.ByteCount) bool {
	congestionWindow := c.GetCongestionWindow()
	if bytesInFlight >= congestionWindow {
		return true
	}
	availableBytes := congestionWindow - bytesInFlight
	slowStartLimited := c.InSlowStart() && bytesInFlight > congestionWindow/2
	return slowStartLimited || availableBytes <= maxBurstBytes
}

// BandwidthEstimate returns the current bandwidth estimate
func (c *cubicSender) BandwidthEstimate() Bandwidth {
	srtt := c.rttStats.SmoothedRTT()
	if srtt == 0 {
		// If we haven't measured an rtt, the bandwidth estimate is unknown.
		return 0
	}
	return BandwidthFromDelta(c.GetCongestionWindow(), srtt)
}

// HybridSlowStart returns the hybrid slow start instance for testing
func (c *cubicSender) HybridSlowStart() *HybridSlowStart {
	return &c.hybridSlowStart
}

// SetNumEmulatedConnections sets the number of emulated connections
func (c *cubicSender) SetNumEmulatedConnections(n int) {
	c.numConnections = utils.Max(n, 1)
	c.cubic.SetNumConnections(c.numConnections)
}

// OnRetransmissionTimeout is called on an retransmission timeout
func (c *cubicSender) OnRetransmissionTimeout(packetsRetransmitted bool) {
	c.largestSentAtLastCutback = protocol.InvalidPacketNumber
	if !packetsRetransmitted {
		return
	}
	c.hybridSlowStart.Restart()
	c.cubic.Reset()
	c.slowstartThreshold = c.congestionWindow / 2
	c.congestionWindow = c.minCongestionWindow
}

// OnConnectionMigration is called when the connection is migrated (?)
func (c *cubicSender) OnConnectionMigration() {
	c.hybridSlowStart.Restart()
	c.largestSentPacketNumber = protocol.InvalidPacketNumber
	c.largestAckedPacketNumber = protocol.InvalidPacketNumber
	c.largestSentAtLastCutback = protocol.InvalidPacketNumber
	c.lastCutbackExitedSlowstart = false
	c.cubic.Reset()
	c.numAckedPackets = 0
	c.congestionWindow = c.initialCongestionWindow
	c.slowstartThreshold = c.initialMaxCongestionWindow
	c.maxCongestionWindow = c.initialMaxCongestionWindow
}

// SetSlowStartLargeReduction allows enabling the SSLR experiment
func (c *cubicSender) SetSlowStartLargeReduction(enabled bool) {
	c.slowStartLargeReduction = enabled
}
