package channels

/*

Implement the processMessages function, it takes a slice of messages.
It should process each message concurrently within a goroutine.
The processing for each message is simple: append -processed to the content.
Use a channel to ensure that all messages are processed and collected correctly
then return the slice of processed messages.

*/
func processMessages(messages []string) []string {
	// ?
	var processed []string
	ch := make(chan string)

	go func() {
		for i := 0; i < len(messages); i++ {
			ch <- messages[i]
		}
	}()

	for i := 0; i < len(messages); i++ {
		msg := <-ch
		processed = append(processed,msg+"-processed")
	}	

	return processed
}
