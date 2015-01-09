package websocket

import (
	"fmt"
	"io"

	ws "code.google.com/p/go.net/websocket"
)

const channelBufSize = 100

var maxId int = 0

type MsgFunc func(c *Client, msg *Message)

// Chat client.
type Client struct {
	id     int
	ws     *ws.Conn
	server *Server
	ch     chan *Message
	doneCh chan bool

	onMessage MsgFunc
}

// Create new chat client.
func NewClient(ws *ws.Conn, server *Server) *Client {

	if ws == nil {
		panic("ws cannot be nil")
	}

	if server == nil {
		panic("server cannot be nil")
	}

	maxId++
	ch := make(chan *Message, channelBufSize)
	doneCh := make(chan bool)

	return &Client{maxId, ws, server, ch, doneCh, nil}
}

func (c *Client) Id() int {
	return c.id
}

func (c *Client) Conn() *ws.Conn {
	return c.ws
}

func (c *Client) Write(data interface{}, id int) {
	c.write(&Message{Id: id, Data: data})
}
func (c *Client) Event(data interface{}, ev string, id int) {
	c.write(&Message{Id: id, Data: data, Event: ev})
}

func (c *Client) write(msg *Message) {
	select {
	case c.ch <- msg:
	default:
		c.server.Del(c)
		err := fmt.Errorf("client %d is disconnected.", c.id)
		c.server.Err(err)
	}
}

func (c *Client) Done() {
	c.doneCh <- true
}

// Listen Write and Read request via chanel
func (c *Client) Listen() {
	go c.listenWrite()
	c.listenRead()
}

// Listen write request via chanel
func (c *Client) listenWrite() {
	for {
		select {

		// send message to the client
		case msg := <-c.ch:
			wslogger.Debugln("Send:", msg)
			ws.JSON.Send(c.ws, msg)

		// receive done request
		case <-c.doneCh:
			c.server.Del(c)
			c.doneCh <- true // for listenRead method
			return
		}
	}
}

// Listen read request via chanel
func (c *Client) listenRead() {
	for {
		select {

		// receive done request
		case <-c.doneCh:
			c.server.Del(c)
			c.doneCh <- true // for listenWrite method
			return

		// read data from ws connection
		default:
			var msg Message
			err := ws.JSON.Receive(c.ws, &msg)
			if err == io.EOF {
				c.doneCh <- true
			} else if err != nil {
				c.server.Err(err)
			} else {
				wslogger.Debugln(&msg)
				if c.onMessage != nil {
					c.onMessage(c, &msg)
				}
			}
		}
	}
}
