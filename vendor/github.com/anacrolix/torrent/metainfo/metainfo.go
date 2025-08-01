package metainfo

import (
	"bufio"
	"fmt"
	"io"
	"net/url"
	"os"
	"time"

	"github.com/anacrolix/torrent/bencode"
	infohash_v2 "github.com/anacrolix/torrent/types/infohash-v2"
)

// Also known as a torrent file.
type MetaInfo struct {
	InfoBytes    bencode.Bytes `bencode:"info,omitempty"`                                                        // BEP 3
	Announce     string        `bencode:"announce,omitempty" mapstructure:",omitempty"`                          // BEP 3
	AnnounceList AnnounceList  `bencode:"announce-list,omitempty"`                                               // BEP 12
	Nodes        []Node        `bencode:"nodes,omitempty,ignore_unmarshal_type_error" mapstructure:",omitempty"` // BEP 5
	// Where's this specified? Mentioned at
	// https://wiki.theory.org/index.php/BitTorrentSpecification: (optional) the creation time of
	// the torrent, in standard UNIX epoch format (integer, seconds since 1-Jan-1970 00:00:00 UTC)
	CreationDate int64   `bencode:"creation date,omitempty,ignore_unmarshal_type_error"`
	Comment      string  `bencode:"comment,omitempty"`
	CreatedBy    string  `bencode:"created by,omitempty"`
	Encoding     string  `bencode:"encoding,omitempty" mapstructure:",omitempty"` // BEP 54
	UrlList      UrlList `bencode:"url-list,omitempty"`                           // BEP 19 WebSeeds
	// BEP 52 (BitTorrent v2): Keys are file merkle roots ("pieces root"s), and the values are the
	// concatenated hashes of the merkle tree layer that corresponds to the piece length.
	PieceLayers map[string]string `bencode:"piece layers,omitempty" mapstructure:",omitempty"`
}

// Load a MetaInfo from an io.Reader. Returns a non-nil error in case of failure.
func Load(r io.Reader) (*MetaInfo, error) {
	var mi MetaInfo
	d := bencode.NewDecoder(r)
	err := d.Decode(&mi)
	if err != nil {
		return nil, err
	}
	err = d.ReadEOF()
	if err != nil {
		err = fmt.Errorf("error after decoding metainfo: %w", err)
	}
	return &mi, err
}

// Convenience function for loading a MetaInfo from a file.
func LoadFromFile(filename string) (*MetaInfo, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var buf bufio.Reader
	buf.Reset(f)
	return Load(&buf)
}

func (mi *MetaInfo) UnmarshalInfo() (info Info, err error) {
	err = bencode.Unmarshal(mi.InfoBytes, &info)
	return
}

func (mi *MetaInfo) HashInfoBytes() (infoHash Hash) {
	return HashBytes(mi.InfoBytes)
}

// Encode to bencoded form.
func (mi *MetaInfo) Write(w io.Writer) error {
	return bencode.NewEncoder(w).Encode(mi)
}

// Set good default values in preparation for creating a new MetaInfo file.
func (mi *MetaInfo) SetDefaults() {
	mi.CreatedBy = "github.com/anacrolix/torrent"
	mi.CreationDate = time.Now().Unix()
}

// Deprecated: Use MagnetV2. Creates a Magnet from a MetaInfo. Optional infohash and parsed info can
// be provided.
func (mi MetaInfo) Magnet(infoHash *Hash, info *Info) (m Magnet) {
	m.Trackers = append(m.Trackers, mi.UpvertedAnnounceList().DistinctValues()...)
	if info != nil {
		m.DisplayName = info.BestName()
	}
	if infoHash != nil {
		m.InfoHash = *infoHash
	} else {
		m.InfoHash = mi.HashInfoBytes()
	}
	m.Params = make(url.Values)
	m.Params["ws"] = mi.UrlList
	return
}

// Creates a MagnetV2 from a MetaInfo. This supports v1, hybrid, and v2 magnet links.
func (mi *MetaInfo) MagnetV2() (m MagnetV2, err error) {
	m.Trackers = append(m.Trackers, mi.UpvertedAnnounceList().DistinctValues()...)
	info, err := mi.UnmarshalInfo()
	if err != nil {
		return
	}
	m.DisplayName = info.BestName()
	if info.HasV1() {
		m.InfoHash.Set(mi.HashInfoBytes())
	}
	if info.HasV2() {
		m.V2InfoHash.Set(infohash_v2.HashBytes(mi.InfoBytes))
	}
	m.Params = make(url.Values)
	m.Params["ws"] = mi.UrlList
	return
}

// Returns the announce-list converted from the old single announce field if necessary.
func (mi *MetaInfo) UpvertedAnnounceList() AnnounceList {
	if mi.AnnounceList.OverridesAnnounce(mi.Announce) {
		return mi.AnnounceList
	}
	if mi.Announce != "" {
		return [][]string{{mi.Announce}}
	}
	return nil
}
