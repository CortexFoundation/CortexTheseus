module github.com/CortexFoundation/CortexTheseus

go 1.24.4

require (
	github.com/Azure/azure-sdk-for-go/sdk/storage/azblob v1.6.2
	github.com/CortexFoundation/inference v1.0.2-0.20230307032835-9197d586a4e8
	github.com/CortexFoundation/statik v0.0.0-20210315012922-8bb8a7b5dc66
	github.com/CortexFoundation/torrentfs v1.0.71-0.20250730095202-a84ecb731cba
	github.com/Microsoft/go-winio v0.6.2
	github.com/VictoriaMetrics/fastcache v1.12.5
	github.com/arsham/figurine v1.3.0
	github.com/aws/aws-sdk-go-v2 v1.37.0
	github.com/aws/aws-sdk-go-v2/config v1.30.1
	github.com/aws/aws-sdk-go-v2/credentials v1.18.1
	github.com/aws/aws-sdk-go-v2/service/route53 v1.54.0
	github.com/cespare/cp v1.1.1
	github.com/charmbracelet/bubbletea v1.3.6
	github.com/cloudflare/cloudflare-go v0.115.0
	github.com/cockroachdb/pebble v1.1.5
	github.com/consensys/gnark-crypto v0.18.0
	github.com/crate-crypto/go-eth-kzg v1.3.0
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc
	github.com/deckarep/golang-set/v2 v2.8.0
	github.com/decred/dcrd/dcrec/secp256k1/v4 v4.4.0
	github.com/dop251/goja v0.0.0-20250630131328-58d95d85e994
	github.com/ethereum/c-kzg-4844 v1.0.3
	github.com/ethereum/go-ethereum v1.16.1
	github.com/ethereum/go-verkle v0.2.2
	github.com/ferranbt/fastssz v0.1.4
	github.com/fjl/gencodec v0.1.0
	github.com/fogleman/ease v0.0.0-20170301025033-8da417bf1776
	github.com/fsnotify/fsnotify v1.9.0
	github.com/gballet/go-libpcsclite v0.0.0-20191108122812-4678299bea08
	github.com/gofrs/flock v0.12.1
	github.com/golang/snappy v1.0.0
	github.com/google/gofuzz v1.2.1-0.20210524182514-9eed411d8615
	github.com/google/uuid v1.6.0
	github.com/gorilla/websocket v1.5.3
	github.com/hashicorp/golang-lru v1.0.2
	github.com/holiman/bloomfilter/v2 v2.0.3
	github.com/holiman/uint256 v1.3.2
	github.com/huin/goupnp v1.3.0
	github.com/influxdata/influxdb-client-go/v2 v2.14.0
	github.com/influxdata/influxdb1-client v0.0.0-20220302092344-a9ab5670611c
	github.com/jackpal/go-nat-pmp v1.0.2
	github.com/jedisct1/go-minisign v0.0.0-20241212093149-d2f9f49435c7
	github.com/lucasb-eyer/go-colorful v1.2.0
	github.com/mattn/go-colorable v0.1.14
	github.com/mattn/go-isatty v0.0.20
	github.com/muesli/reflow v0.3.0
	github.com/muesli/termenv v0.16.0
	github.com/naoina/toml v0.1.2-0.20170918210437-9fafd6967416
	github.com/olekukonko/tablewriter v1.0.9
	github.com/peterh/liner v1.2.2
	github.com/pion/stun/v2 v2.0.0
	github.com/rs/cors v1.11.1
	github.com/shirou/gopsutil v3.21.11+incompatible
	github.com/status-im/keycard-go v0.3.3
	github.com/steakknife/bloomfilter v0.0.0-20180922174646-6819c0d2a570
	github.com/stretchr/testify v1.10.0
	github.com/syndtr/goleveldb v1.0.1-0.20220721030215-126854af5e6d
	github.com/ucwong/color v1.10.1-0.20200624105241-fba1e010fe1e
	github.com/urfave/cli/v2 v2.27.7
	go.uber.org/automaxprocs v1.6.0
	golang.org/x/crypto v0.40.0
	golang.org/x/mobile v0.0.0-20201217150744-e6ae53a27f4f
	golang.org/x/sync v0.16.0
	golang.org/x/sys v0.34.0
	golang.org/x/text v0.27.0
	golang.org/x/time v0.12.0
	golang.org/x/tools v0.35.0
	google.golang.org/protobuf v1.36.6
	gopkg.in/check.v1 v1.0.0-20201130134442-10cb98267c6c
	gopkg.in/urfave/cli.v1 v1.20.0
)

require (
	github.com/Azure/azure-sdk-for-go/sdk/azcore v1.18.1 // indirect
	github.com/Azure/azure-sdk-for-go/sdk/internal v1.11.1 // indirect
	github.com/CortexFoundation/compress v0.0.0-20240218153512-9074bdc2397c // indirect
	github.com/CortexFoundation/cvm-runtime v0.0.0-20221117094012-b5a251885572 // indirect
	github.com/CortexFoundation/merkletree v0.0.0-20240407093305-416581a62b5b // indirect
	github.com/CortexFoundation/robot v1.0.7-0.20250720133840-49217aa96f7e // indirect
	github.com/CortexFoundation/wormhole v0.0.2-0.20250730094041-d3c94276bb65 // indirect
	github.com/DataDog/zstd v1.5.7 // indirect
	github.com/RoaringBitmap/roaring v1.9.4 // indirect
	github.com/ajwerner/btree v0.0.0-20211221152037-f427b3e689c0 // indirect
	github.com/alecthomas/atomic v0.1.0-alpha2 // indirect
	github.com/anacrolix/chansync v0.6.0 // indirect
	github.com/anacrolix/dht/v2 v2.22.2-0.20250623060212-d7b7d8a52b01 // indirect
	github.com/anacrolix/envpprof v1.4.0 // indirect
	github.com/anacrolix/generics v0.0.3 // indirect
	github.com/anacrolix/go-libutp v1.3.2 // indirect
	github.com/anacrolix/log v0.16.1-0.20250526073428-5cb74e15092b // indirect
	github.com/anacrolix/missinggo v1.3.0 // indirect
	github.com/anacrolix/missinggo/perf v1.0.0 // indirect
	github.com/anacrolix/missinggo/v2 v2.9.0 // indirect
	github.com/anacrolix/mmsg v1.1.1 // indirect
	github.com/anacrolix/multiless v0.4.0 // indirect
	github.com/anacrolix/stm v0.5.0 // indirect
	github.com/anacrolix/sync v0.5.4 // indirect
	github.com/anacrolix/torrent v1.58.2-0.20250720014114-dda1d97c6a22 // indirect
	github.com/anacrolix/upnp v0.1.4 // indirect
	github.com/anacrolix/utp v0.2.0 // indirect
	github.com/antlabs/stl v0.0.2 // indirect
	github.com/antlabs/timer v0.1.4 // indirect
	github.com/apapsch/go-jsonmerge/v2 v2.0.0 // indirect
	github.com/arsham/rainbow v1.2.1 // indirect
	github.com/aws/aws-sdk-go-v2/feature/ec2/imds v1.18.0 // indirect
	github.com/aws/aws-sdk-go-v2/internal/configsources v1.4.0 // indirect
	github.com/aws/aws-sdk-go-v2/internal/endpoints/v2 v2.7.0 // indirect
	github.com/aws/aws-sdk-go-v2/internal/ini v1.8.3 // indirect
	github.com/aws/aws-sdk-go-v2/service/internal/accept-encoding v1.13.0 // indirect
	github.com/aws/aws-sdk-go-v2/service/internal/presigned-url v1.13.0 // indirect
	github.com/aws/aws-sdk-go-v2/service/sso v1.26.0 // indirect
	github.com/aws/aws-sdk-go-v2/service/ssooidc v1.31.0 // indirect
	github.com/aws/aws-sdk-go-v2/service/sts v1.35.0 // indirect
	github.com/aws/smithy-go v1.22.5 // indirect
	github.com/aymanbagabas/go-osc52/v2 v2.0.1 // indirect
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/benbjohnson/immutable v0.4.3 // indirect
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/bits-and-blooms/bitset v1.22.0 // indirect
	github.com/bradfitz/iter v0.0.0-20191230175014-e8f45d346db8 // indirect
	github.com/bwmarrin/snowflake v0.3.0 // indirect
	github.com/cespare/xxhash v1.1.0 // indirect
	github.com/cespare/xxhash/v2 v2.3.0 // indirect
	github.com/charmbracelet/colorprofile v0.3.1 // indirect
	github.com/charmbracelet/lipgloss v1.1.0 // indirect
	github.com/charmbracelet/x/ansi v0.9.3 // indirect
	github.com/charmbracelet/x/cellbuf v0.0.13 // indirect
	github.com/charmbracelet/x/term v0.2.1 // indirect
	github.com/cockroachdb/errors v1.12.0 // indirect
	github.com/cockroachdb/fifo v0.0.0-20240816210425-c5d0cb0b6fc0 // indirect
	github.com/cockroachdb/logtags v0.0.0-20241215232642-bb51bb14a506 // indirect
	github.com/cockroachdb/redact v1.1.6 // indirect
	github.com/cockroachdb/tokenbucket v0.0.0-20250429170803-42689b6311bb // indirect
	github.com/common-nighthawk/go-figure v0.0.0-20210622060536-734e95fb86be // indirect
	github.com/cpuguy83/go-md2man/v2 v2.0.7 // indirect
	github.com/crate-crypto/go-ipa v0.0.0-20240724233137-53bbb0ceb27a // indirect
	github.com/dgraph-io/badger/v4 v4.8.0 // indirect
	github.com/dgraph-io/ristretto/v2 v2.2.0 // indirect
	github.com/dlclark/regexp2 v1.11.5 // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/edsrzf/mmap-go v1.2.0 // indirect
	github.com/elliotchance/orderedmap v1.8.0 // indirect
	github.com/emicklei/dot v1.8.0 // indirect
	github.com/erikgeiser/coninput v0.0.0-20211004153227-1c3628e74d0f // indirect
	github.com/fatih/color v1.18.0 // indirect
	github.com/garslo/gogen v0.0.0-20170306192744-1d203ffc1f61 // indirect
	github.com/getsentry/sentry-go v0.34.1 // indirect
	github.com/go-llsqlite/adapter v0.2.0 // indirect
	github.com/go-llsqlite/crawshaw v0.6.0 // indirect
	github.com/go-logr/logr v1.4.3 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/go-ole/go-ole v1.3.0 // indirect
	github.com/go-resty/resty/v2 v2.16.5 // indirect
	github.com/go-sourcemap/sourcemap v2.1.4+incompatible // indirect
	github.com/goccy/go-json v0.10.5 // indirect
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/google/btree v1.1.3 // indirect
	github.com/google/flatbuffers v25.2.10+incompatible // indirect
	github.com/google/go-querystring v1.1.0 // indirect
	github.com/google/pprof v0.0.0-20250630185457-6e76a2b096b5 // indirect
	github.com/hashicorp/golang-lru/v2 v2.0.7 // indirect
	github.com/huandu/xstrings v1.5.0 // indirect
	github.com/influxdata/line-protocol v0.0.0-20210922203350-b1ad95c89adf // indirect
	github.com/jedib0t/go-pretty/v6 v6.6.8 // indirect
	github.com/klauspost/compress v1.18.0 // indirect
	github.com/klauspost/cpuid/v2 v2.3.0 // indirect
	github.com/kr/pretty v0.3.1 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/mattn/go-localereader v0.0.1 // indirect
	github.com/mattn/go-runewidth v0.0.16 // indirect
	github.com/minio/sha256-simd v1.0.1 // indirect
	github.com/mitchellh/mapstructure v1.5.0 // indirect
	github.com/mr-tron/base58 v1.2.0 // indirect
	github.com/mschoch/smat v0.2.0 // indirect
	github.com/muesli/ansi v0.0.0-20230316100256-276c6243b2f6 // indirect
	github.com/muesli/cancelreader v0.2.2 // indirect
	github.com/multiformats/go-multihash v0.2.3 // indirect
	github.com/multiformats/go-varint v0.0.7 // indirect
	github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822 // indirect
	github.com/naoina/go-stringutil v0.1.0 // indirect
	github.com/ncruces/go-strftime v0.1.9 // indirect
	github.com/nutsdb/nutsdb v1.0.4 // indirect
	github.com/oapi-codegen/runtime v1.1.2 // indirect
	github.com/olekukonko/errors v1.1.0 // indirect
	github.com/olekukonko/ll v0.0.9 // indirect
	github.com/otiai10/copy v1.14.1 // indirect
	github.com/otiai10/mint v1.6.3 // indirect
	github.com/pion/datachannel v1.5.10 // indirect
	github.com/pion/dtls/v2 v2.2.12 // indirect
	github.com/pion/dtls/v3 v3.0.6 // indirect
	github.com/pion/ice/v4 v4.0.10 // indirect
	github.com/pion/interceptor v0.1.40 // indirect
	github.com/pion/logging v0.2.4 // indirect
	github.com/pion/mdns/v2 v2.0.7 // indirect
	github.com/pion/randutil v0.1.0 // indirect
	github.com/pion/rtcp v1.2.15 // indirect
	github.com/pion/rtp v1.8.21 // indirect
	github.com/pion/sctp v1.8.39 // indirect
	github.com/pion/sdp/v3 v3.0.15 // indirect
	github.com/pion/srtp/v3 v3.0.6 // indirect
	github.com/pion/stun/v3 v3.0.0 // indirect
	github.com/pion/transport/v2 v2.2.10 // indirect
	github.com/pion/transport/v3 v3.0.7 // indirect
	github.com/pion/turn/v4 v4.0.2 // indirect
	github.com/pion/webrtc/v4 v4.1.3 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/prometheus/client_golang v1.22.0 // indirect
	github.com/prometheus/client_model v0.6.2 // indirect
	github.com/prometheus/common v0.65.0 // indirect
	github.com/prometheus/procfs v0.17.0 // indirect
	github.com/protolambda/ctxlock v0.1.0 // indirect
	github.com/rakyll/statik v0.1.7 // indirect
	github.com/remyoudompheng/bigfft v0.0.0-20230129092748-24d4a6f8daec // indirect
	github.com/rivo/uniseg v0.4.7 // indirect
	github.com/rogpeppe/go-internal v1.14.1 // indirect
	github.com/rs/dnscache v0.0.0-20230804202142-fc85eb664529 // indirect
	github.com/russross/blackfriday/v2 v2.1.0 // indirect
	github.com/spaolacci/murmur3 v1.1.0 // indirect
	github.com/steakknife/hamming v0.0.0-20180906055917-c99c65617cd3 // indirect
	github.com/supranational/blst v0.3.15 // indirect
	github.com/tidwall/btree v1.8.0 // indirect
	github.com/tidwall/hashmap v1.8.1 // indirect
	github.com/tklauser/go-sysconf v0.3.15 // indirect
	github.com/tklauser/numcpus v0.10.0 // indirect
	github.com/ucwong/filecache v1.0.6-0.20230405163841-810d53ced4bd // indirect
	github.com/ucwong/go-ttlmap v1.0.2-0.20221020173635-331e7ddde2bb // indirect
	github.com/ucwong/golang-kv v1.0.24-0.20250720145037-4c149701f3b8 // indirect
	github.com/ucwong/shard v1.0.1-0.20250426172507-f1db2901f62c // indirect
	github.com/wlynxg/anet v0.0.5 // indirect
	github.com/xo/terminfo v0.0.0-20220910002029-abceb7e1c41e // indirect
	github.com/xrash/smetrics v0.0.0-20250705151800-55b8f293f342 // indirect
	github.com/xujiajun/mmap-go v1.0.1 // indirect
	github.com/xujiajun/utils v0.0.0-20220904132955-5f7c5b914235 // indirect
	github.com/yusufpapurcu/wmi v1.2.4 // indirect
	github.com/zeebo/xxh3 v1.0.3-0.20230502181907-3808c706a06a // indirect
	go.etcd.io/bbolt v1.4.2 // indirect
	go.opentelemetry.io/auto/sdk v1.1.0 // indirect
	go.opentelemetry.io/otel v1.37.0 // indirect
	go.opentelemetry.io/otel/metric v1.37.0 // indirect
	go.opentelemetry.io/otel/trace v1.37.0 // indirect
	golang.org/x/exp v0.0.0-20250718183923-645b1fa84792 // indirect
	golang.org/x/mod v0.26.0 // indirect
	golang.org/x/net v0.42.0 // indirect
	golang.org/x/term v0.33.0 // indirect
	gopkg.in/yaml.v2 v2.4.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	lukechampine.com/blake3 v1.4.1 // indirect
	modernc.org/libc v1.66.6 // indirect
	modernc.org/mathutil v1.7.1 // indirect
	modernc.org/memory v1.11.0 // indirect
	modernc.org/sqlite v1.38.2 // indirect
	zombiezen.com/go/sqlite v1.4.2 // indirect
)
