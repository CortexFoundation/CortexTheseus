package torrentfs

import (
	"context"
	"time"
)

func (api *PublicTorrentAPI) Version(ctx context.Context) string {
	return ProtocolVersionStr
}

type PublicTorrentAPI struct {
	w *TorrentFS

	lastUsed map[string]time.Time // keeps track when a filter was polled for the last time.
}

func NewPublicTorrentAPI(w *TorrentFS) *PublicTorrentAPI {
	api := &PublicTorrentAPI{
		w:        w,
		lastUsed: make(map[string]time.Time),
	}
	return api
}
