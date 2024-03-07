package metainfo

import (
	g "github.com/anacrolix/generics"
)

type Piece struct {
	Info *Info // Can we embed the fields here instead, or is it something to do with saving memory?
	i    pieceIndex
}

type pieceIndex = int

func (p Piece) Length() int64 {
	if p.Info.HasV2() {
		var offset int64
		pieceLength := p.Info.PieceLength
		lastFileEnd := int64(0)
		done := false
		p.Info.FileTree.upvertedFiles(pieceLength, func(fi FileInfo) {
			if done {
				return
			}
			fileStartPiece := int(offset / pieceLength)
			if fileStartPiece > p.i {
				done = true
				return
			}
			lastFileEnd = offset + fi.Length
			offset = (lastFileEnd + pieceLength - 1) / pieceLength * pieceLength
		})
		ret := min(lastFileEnd-int64(p.i)*pieceLength, pieceLength)
		if ret <= 0 {
			panic(ret)
		}
		return ret
	}
	if p.i == p.Info.NumPieces()-1 {
		return p.Info.TotalLength() - int64(p.i)*p.Info.PieceLength
	}
	return p.Info.PieceLength
}

func (p Piece) Offset() int64 {
	return int64(p.i) * p.Info.PieceLength
}

func (p Piece) V1Hash() (ret g.Option[Hash]) {
	if !p.Info.HasV1() {
		return
	}
	copy(ret.Value[:], p.Info.Pieces[p.i*HashSize:(p.i+1)*HashSize])
	ret.Ok = true
	return
}

func (p Piece) Index() int {
	return p.i
}
