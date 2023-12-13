// Copyright 2023 The CortexTheseus Authors
// This file is part of The CortexTheseus library.
//
// The CortexTheseus library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// The CortexTheseus library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with The CortexTheseus library. If not, see <http://www.gnu.org/licenses/>.

package vm

func memoryInfer(stack *Stack) (uint64, bool) {
	return calcMemSize64(stack.Back(2), stack.Back(3))
}
