package main

import (
	"context"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"

	"github.com/anacrolix/dht/v2"
	_ "github.com/anacrolix/envpprof"
	"github.com/anacrolix/tagflag"
)

var (
	flags = struct {
		TableFile   string `help:"name of file for storing node info"`
		Addr        string `help:"local UDP address"`
		NoBootstrap bool
	}{
		Addr: ":0",
	}
	s *dht.Server
)

func loadTable() (err error) {
	added, err := s.AddNodesFromFile(flags.TableFile)
	log.Printf("loaded %d nodes from table file", added)
	return
}

func saveTable() error {
	return dht.WriteNodesToFile(s.Nodes(), flags.TableFile)
}

func main() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	tagflag.Parse(&flags)
	conn, err := net.ListenPacket("udp", flags.Addr)
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()
	s, err = dht.NewServer(&dht.ServerConfig{
		Conn:          conn,
		StartingNodes: dht.GlobalBootstrapAddrs,
	})
	if err != nil {
		log.Fatal(err)
	}
	http.HandleFunc("/debug/dht", func(w http.ResponseWriter, r *http.Request) {
		s.WriteStatus(w)
	})
	if flags.TableFile != "" {
		err = loadTable()
		if err != nil {
			log.Fatalf("error loading table: %s", err)
		}
	}
	log.Printf("dht server on %s, ID is %x", s.Addr(), s.ID())

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		ch := make(chan os.Signal, 1)
		signal.Notify(ch)
		<-ch
		cancel()
	}()
	if !flags.NoBootstrap {
		go func() {
			if tried, err := s.Bootstrap(); err != nil {
				log.Printf("error bootstrapping: %s", err)
			} else {
				log.Printf("finished bootstrapping: crawled %#v addrs", tried)
			}
		}()
	}
	<-ctx.Done()
	s.Close()

	if flags.TableFile != "" {
		if err := saveTable(); err != nil {
			log.Printf("error saving node table: %s", err)
		}
	}
}
