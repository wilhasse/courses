package channels

/*

Correct implementation each message in a separeted go routine with synchorization

Implement the processMessages function, it takes a slice of messages.
It should process each message concurrently within a goroutine.
The processing for each message is simple: append -processed to the content.
Use a channel to ensure that all messages are processed and collected ]
correctly then return the slice of processed messages.

*/

import (
	"sync"
)

func processMessages(messages []string) []string {
	var wg sync.WaitGroup
	ch := make(chan string, len(messages))

	// Launch a goroutine for each message
	for _, msg := range messages {
		wg.Add(1) // Increment the WaitGroup counter
		go func(m string) {
			// Decrement the counter when the goroutine completes
			defer wg.Done()
			processedMessage := m + "-processed"
			ch <- processedMessage
		}(msg)
	}

	// Launch a goroutine to close the channel once all messages are processed
	go func() {
		// Wait for all goroutines to finish
		wg.Wait()
		// Close the channel to signal that no more messages will be sent
		close(ch)
	}()

	// Collect all processed messages from the channel
	var processed []string
	// Range over the channel until it's closed
	for pmsg := range ch {
		processed = append(processed, pmsg)
	}

	return processed
}
