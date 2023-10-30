module github.com/CortexFoundation/CortexTheseus

go 1.21

require (
	github.com/Azure/azure-sdk-for-go/sdk/storage/azblob v0.4.1
	github.com/CortexFoundation/inference v1.0.2-0.20230307032835-9197d586a4e8
	github.com/CortexFoundation/statik v0.0.0-20210315012922-8bb8a7b5dc66
	github.com/CortexFoundation/torrentfs v1.0.55-0.20231027203605-3f8792552f00
	github.com/VictoriaMetrics/fastcache v1.12.1
	github.com/arsham/figurine v1.3.0
	github.com/aws/aws-sdk-go-v2 v1.21.2
	github.com/aws/aws-sdk-go-v2/config v1.19.0
	github.com/aws/aws-sdk-go-v2/credentials v1.13.43
	github.com/aws/aws-sdk-go-v2/service/route53 v1.30.2
	github.com/btcsuite/btcd/btcec/v2 v2.3.2
	github.com/cespare/cp v1.1.1
	github.com/charmbracelet/bubbletea v0.24.2
	github.com/cloudflare/cloudflare-go v0.57.1
	github.com/cockroachdb/errors v1.11.1
	github.com/cockroachdb/pebble v0.0.0-20231027194153-ed45a7767175
	github.com/consensys/gnark-crypto v0.12.1
	github.com/crate-crypto/go-kzg-4844 v0.7.0
	github.com/davecgh/go-spew v1.1.2-0.20180830191138-d8f796af33cc
	github.com/deckarep/golang-set/v2 v2.3.1
	github.com/dop251/goja v0.0.0-20231024180952-594410467bc6
	github.com/ethereum/c-kzg-4844 v0.4.0
	github.com/fjl/gencodec v0.0.0-20230517082657-f9840df7b83e
	github.com/fjl/memsize v0.0.1
	github.com/fogleman/ease v0.0.0-20170301025033-8da417bf1776
	github.com/fsnotify/fsnotify v1.7.0
	github.com/gballet/go-libpcsclite v0.0.0-20191108122812-4678299bea08
	github.com/gballet/go-verkle v0.0.0-20230620072649-dd771ce9a3b3
	github.com/go-stack/stack v1.8.1
	github.com/gofrs/flock v0.8.1
	github.com/golang/snappy v0.0.5-0.20220116011046-fa5810519dcb
	github.com/google/gofuzz v1.2.1-0.20210524182514-9eed411d8615
	github.com/google/uuid v1.3.1
	github.com/gorilla/websocket v1.5.0
	github.com/hashicorp/golang-lru v1.0.2
	github.com/holiman/bloomfilter/v2 v2.0.3
	github.com/holiman/uint256 v1.2.3
	github.com/huin/goupnp v1.3.0
	github.com/influxdata/influxdb-client-go/v2 v2.12.3
	github.com/influxdata/influxdb1-client v0.0.0-20220302092344-a9ab5670611c
	github.com/jackpal/go-nat-pmp v1.0.2
	github.com/jedisct1/go-minisign v0.0.0-20230811132847-661be99b8267
	github.com/julienschmidt/httprouter v1.3.0
	github.com/lucasb-eyer/go-colorful v1.2.0
	github.com/mattn/go-colorable v0.1.13
	github.com/mattn/go-isatty v0.0.20
	github.com/muesli/reflow v0.3.0
	github.com/muesli/termenv v0.15.2
	github.com/naoina/toml v0.1.2-0.20170918210437-9fafd6967416
	github.com/olekukonko/tablewriter v0.0.5
	github.com/peterh/liner v1.2.2
	github.com/rs/cors v1.10.1
	github.com/shirou/gopsutil v3.21.11+incompatible
	github.com/status-im/keycard-go v0.3.2
	github.com/steakknife/bloomfilter v0.0.0-20180922174646-6819c0d2a570
	github.com/stretchr/testify v1.8.4
	github.com/syndtr/goleveldb v1.0.1-0.20220721030215-126854af5e6d
	github.com/ucwong/color v1.10.1-0.20200624105241-fba1e010fe1e
	github.com/urfave/cli/v2 v2.25.7
	go.uber.org/automaxprocs v1.5.3
	golang.org/x/crypto v0.14.0
	golang.org/x/exp v0.0.0-20231006140011-7918f672742d
	golang.org/x/mobile v0.0.0-20201217150744-e6ae53a27f4f
	golang.org/x/sync v0.4.0
	golang.org/x/sys v0.13.0
	golang.org/x/text v0.13.0
	golang.org/x/time v0.3.0
	golang.org/x/tools v0.14.0
	google.golang.org/protobuf v1.31.0
	gopkg.in/check.v1 v1.0.0-20201130134442-10cb98267c6c
	gopkg.in/natefinch/npipe.v2 v2.0.0-20160621034901-c1b8fa8bdcce
	gopkg.in/urfave/cli.v1 v1.20.0
)

require (
	github.com/Azure/azure-sdk-for-go/sdk/azcore v1.0.0 // indirect
	github.com/Azure/azure-sdk-for-go/sdk/internal v1.0.0 // indirect
	github.com/CortexFoundation/cvm-runtime v0.0.0-20221117094012-b5a251885572 // indirect
	github.com/CortexFoundation/merkletree v0.0.0-20230724124840-b6e80265a137 // indirect
	github.com/CortexFoundation/robot v1.0.7-0.20231001204802-6299a9ffd845 // indirect
	github.com/CortexFoundation/wormhole v0.0.2-0.20231018202213-693acd0cc941 // indirect
	github.com/DataDog/zstd v1.5.6-0.20230622172052-ea68dcab66c0 // indirect
	github.com/RoaringBitmap/roaring v1.3.0 // indirect
	github.com/ajwerner/btree v0.0.0-20211221152037-f427b3e689c0 // indirect
	github.com/alecthomas/atomic v0.1.0-alpha2 // indirect
	github.com/anacrolix/chansync v0.3.0 // indirect
	github.com/anacrolix/dht/v2 v2.20.0 // indirect
	github.com/anacrolix/envpprof v1.3.0 // indirect
	github.com/anacrolix/generics v0.0.0-20230816105729-c755655aee45 // indirect
	github.com/anacrolix/go-libutp v1.3.1 // indirect
	github.com/anacrolix/log v0.14.3-0.20230823030427-4b296d71a6b4 // indirect
	github.com/anacrolix/missinggo v1.3.0 // indirect
	github.com/anacrolix/missinggo/perf v1.0.0 // indirect
	github.com/anacrolix/missinggo/v2 v2.7.2 // indirect
	github.com/anacrolix/mmsg v1.0.0 // indirect
	github.com/anacrolix/multiless v0.3.1-0.20221221005021-2d12701f83f7 // indirect
	github.com/anacrolix/stm v0.5.0 // indirect
	github.com/anacrolix/sync v0.5.1 // indirect
	github.com/anacrolix/torrent v1.53.1 // indirect
	github.com/anacrolix/upnp v0.1.3-0.20220123035249-922794e51c96 // indirect
	github.com/anacrolix/utp v0.2.0 // indirect
	github.com/antlabs/stl v0.0.1 // indirect
	github.com/antlabs/timer v0.0.11 // indirect
	github.com/arsham/rainbow v1.2.1 // indirect
	github.com/aws/aws-sdk-go-v2/feature/ec2/imds v1.13.13 // indirect
	github.com/aws/aws-sdk-go-v2/internal/configsources v1.1.43 // indirect
	github.com/aws/aws-sdk-go-v2/internal/endpoints/v2 v2.4.37 // indirect
	github.com/aws/aws-sdk-go-v2/internal/ini v1.3.45 // indirect
	github.com/aws/aws-sdk-go-v2/service/internal/presigned-url v1.9.37 // indirect
	github.com/aws/aws-sdk-go-v2/service/sso v1.15.2 // indirect
	github.com/aws/aws-sdk-go-v2/service/ssooidc v1.17.3 // indirect
	github.com/aws/aws-sdk-go-v2/service/sts v1.23.2 // indirect
	github.com/aws/smithy-go v1.15.0 // indirect
	github.com/aymanbagabas/go-osc52/v2 v2.0.1 // indirect
	github.com/bahlo/generic-list-go v0.2.0 // indirect
	github.com/benbjohnson/immutable v0.4.3 // indirect
	github.com/beorn7/perks v1.0.1 // indirect
	github.com/bits-and-blooms/bitset v1.8.0 // indirect
	github.com/bradfitz/iter v0.0.0-20191230175014-e8f45d346db8 // indirect
	github.com/bwmarrin/snowflake v0.3.0 // indirect
	github.com/cespare/xxhash/v2 v2.2.0 // indirect
	github.com/cockroachdb/logtags v0.0.0-20230118201751-21c54148d20b // indirect
	github.com/cockroachdb/redact v1.1.5 // indirect
	github.com/cockroachdb/tokenbucket v0.0.0-20230807174530-cc333fc44b06 // indirect
	github.com/common-nighthawk/go-figure v0.0.0-20210622060536-734e95fb86be // indirect
	github.com/consensys/bavard v0.1.13 // indirect
	github.com/containerd/console v1.0.4-0.20230313162750-1ae8d489ac81 // indirect
	github.com/cpuguy83/go-md2man/v2 v2.0.2 // indirect
	github.com/crate-crypto/go-ipa v0.0.0-20230601170251-1830d0757c80 // indirect
	github.com/decred/dcrd/dcrec/secp256k1/v4 v4.2.0 // indirect
	github.com/deepmap/oapi-codegen v1.8.2 // indirect
	github.com/dgraph-io/badger/v4 v4.2.1-0.20231013074411-fb1b00959581 // indirect
	github.com/dgraph-io/ristretto v0.1.1 // indirect
	github.com/dlclark/regexp2 v1.7.0 // indirect
	github.com/dustin/go-humanize v1.0.1 // indirect
	github.com/edsrzf/mmap-go v1.1.1-0.20220903035803-8e5d0fe06024 // indirect
	github.com/elliotchance/orderedmap v1.5.0 // indirect
	github.com/garslo/gogen v0.0.0-20170306192744-1d203ffc1f61 // indirect
	github.com/getsentry/sentry-go v0.22.0 // indirect
	github.com/go-llsqlite/adapter v0.0.0-20230927005056-7f5ce7f0c916 // indirect
	github.com/go-llsqlite/crawshaw v0.4.0 // indirect
	github.com/go-logr/logr v1.2.4 // indirect
	github.com/go-logr/stdr v1.2.2 // indirect
	github.com/go-ole/go-ole v1.2.6 // indirect
	github.com/go-resty/resty/v2 v2.10.0 // indirect
	github.com/go-sourcemap/sourcemap v2.1.3+incompatible // indirect
	github.com/gogo/protobuf v1.3.2 // indirect
	github.com/golang/glog v1.1.1 // indirect
	github.com/golang/groupcache v0.0.0-20210331224755-41bb18bfe9da // indirect
	github.com/golang/protobuf v1.5.3 // indirect
	github.com/google/btree v1.1.2 // indirect
	github.com/google/flatbuffers v23.5.26+incompatible // indirect
	github.com/google/go-querystring v1.1.0 // indirect
	github.com/google/pprof v0.0.0-20230207041349-798e818bf904 // indirect
	github.com/hashicorp/go-cleanhttp v0.5.2 // indirect
	github.com/hashicorp/go-retryablehttp v0.7.1 // indirect
	github.com/hashicorp/golang-lru/v2 v2.0.7 // indirect
	github.com/huandu/xstrings v1.4.0 // indirect
	github.com/influxdata/line-protocol v0.0.0-20200327222509-2487e7298839 // indirect
	github.com/jedib0t/go-pretty/v6 v6.4.6 // indirect
	github.com/jmespath/go-jmespath v0.4.0 // indirect
	github.com/klauspost/compress v1.16.7 // indirect
	github.com/klauspost/cpuid/v2 v2.2.5 // indirect
	github.com/kr/pretty v0.3.1 // indirect
	github.com/kr/text v0.2.0 // indirect
	github.com/mattn/go-localereader v0.0.1 // indirect
	github.com/mattn/go-runewidth v0.0.15 // indirect
	github.com/matttproud/golang_protobuf_extensions v1.0.4 // indirect
	github.com/mmcloughlin/addchain v0.4.0 // indirect
	github.com/mschoch/smat v0.2.0 // indirect
	github.com/muesli/ansi v0.0.0-20211018074035-2e021307bc4b // indirect
	github.com/muesli/cancelreader v0.2.2 // indirect
	github.com/naoina/go-stringutil v0.1.0 // indirect
	github.com/nutsdb/nutsdb v0.14.2-0.20231027074712-e6f23063dbc6 // indirect
	github.com/otiai10/copy v1.12.0 // indirect
	github.com/pion/datachannel v1.5.5 // indirect
	github.com/pion/dtls/v2 v2.2.7 // indirect
	github.com/pion/ice/v2 v2.3.9 // indirect
	github.com/pion/interceptor v0.1.17 // indirect
	github.com/pion/logging v0.2.2 // indirect
	github.com/pion/mdns v0.0.7 // indirect
	github.com/pion/randutil v0.1.0 // indirect
	github.com/pion/rtcp v1.2.10 // indirect
	github.com/pion/rtp v1.8.0 // indirect
	github.com/pion/sctp v1.8.7 // indirect
	github.com/pion/sdp/v3 v3.0.6 // indirect
	github.com/pion/srtp/v2 v2.0.15 // indirect
	github.com/pion/stun v0.6.1 // indirect
	github.com/pion/transport/v2 v2.2.1 // indirect
	github.com/pion/turn/v2 v2.1.3 // indirect
	github.com/pion/webrtc/v3 v3.2.13 // indirect
	github.com/pkg/errors v0.9.1 // indirect
	github.com/pmezard/go-difflib v1.0.0 // indirect
	github.com/prometheus/client_golang v1.16.0 // indirect
	github.com/prometheus/client_model v0.4.0 // indirect
	github.com/prometheus/common v0.44.0 // indirect
	github.com/prometheus/procfs v0.11.0 // indirect
	github.com/rakyll/statik v0.1.7 // indirect
	github.com/remyoudompheng/bigfft v0.0.0-20230129092748-24d4a6f8daec // indirect
	github.com/rivo/uniseg v0.4.4 // indirect
	github.com/rogpeppe/go-internal v1.11.0 // indirect
	github.com/rosedblabs/go-immutable-radix/v2 v2.0.1-0.20230614125820-f2a7bc058c90 // indirect
	github.com/rosedblabs/rosedb/v2 v2.2.2 // indirect
	github.com/rosedblabs/wal v1.3.1 // indirect
	github.com/rs/dnscache v0.0.0-20211102005908-e0241e321417 // indirect
	github.com/russross/blackfriday/v2 v2.1.0 // indirect
	github.com/steakknife/hamming v0.0.0-20180906055917-c99c65617cd3 // indirect
	github.com/supranational/blst v0.3.11 // indirect
	github.com/tidwall/btree v1.6.0 // indirect
	github.com/tidwall/hashmap v1.8.1 // indirect
	github.com/tklauser/go-sysconf v0.3.11 // indirect
	github.com/tklauser/numcpus v0.6.1 // indirect
	github.com/ucwong/filecache v1.0.6-0.20230405163841-810d53ced4bd // indirect
	github.com/ucwong/go-ttlmap v1.0.2-0.20221020173635-331e7ddde2bb // indirect
	github.com/ucwong/golang-kv v1.0.23-0.20231027202923-47a531158579 // indirect
	github.com/ucwong/shard v1.0.1-0.20230924231639-2ac2d8ab288c // indirect
	github.com/xrash/smetrics v0.0.0-20201216005158-039620a65673 // indirect
	github.com/xujiajun/mmap-go v1.0.1 // indirect
	github.com/xujiajun/utils v0.0.0-20220904132955-5f7c5b914235 // indirect
	github.com/yusufpapurcu/wmi v1.2.3 // indirect
	github.com/zeebo/xxh3 v1.0.3-0.20230502181907-3808c706a06a // indirect
	go.etcd.io/bbolt v1.3.7 // indirect
	go.opencensus.io v0.24.0 // indirect
	go.opentelemetry.io/otel v1.16.0 // indirect
	go.opentelemetry.io/otel/metric v1.16.0 // indirect
	go.opentelemetry.io/otel/trace v1.16.0 // indirect
	golang.org/x/mod v0.13.0 // indirect
	golang.org/x/net v0.17.0 // indirect
	golang.org/x/term v0.13.0 // indirect
	gopkg.in/yaml.v3 v3.0.1 // indirect
	modernc.org/libc v1.22.3 // indirect
	modernc.org/mathutil v1.5.0 // indirect
	modernc.org/memory v1.5.0 // indirect
	modernc.org/sqlite v1.21.1 // indirect
	rsc.io/tmplfunc v0.0.3 // indirect
	zombiezen.com/go/sqlite v0.13.1 // indirect
)
