use super::*;

impl TestOnlyNativeStreamingSession {
    pub(super) fn close_impl(
        &mut self,
        mode: CloseMode,
    ) -> Result<Vec<RealtimeEventEnvelope>, NativeAsrError> {
        if self.closed {
            return Ok(Vec::new());
        }

        self.closed = matches!(mode, CloseMode::Close | CloseMode::Cancel);
        let mut events = self.drain_pending_events();

        match mode {
            CloseMode::Finish | CloseMode::Close => {
                if let Some(update) = self.emitter.take_pending_partial_update() {
                    events.extend(self.emitter.apply_final(update, test_time(89))?);
                }
                self.push_stop_close_events(mode, &mut events)?;
            }
            CloseMode::Cancel => {
                events.extend(self.emitter.cancel(
                    "Test-only native streaming fixture session was cancelled.",
                    test_time(80),
                    test_time(81),
                )?);
            }
        }
        Ok(events)
    }

    fn push_stop_close_events(
        &mut self,
        mode: CloseMode,
        events: &mut Vec<RealtimeEventEnvelope>,
    ) -> Result<(), NativeAsrError> {
        let stop_reason = if matches!(mode, CloseMode::Close) {
            "client_closed"
        } else {
            "input_finished"
        };
        if let Some(stopped) = self.emitter.close_if_running(stop_reason, test_time(90))? {
            events.push(stopped);
        }
        if matches!(mode, CloseMode::Close) {
            events.push(self.emitter.close_session("client_closed", test_time(91))?);
        }
        Ok(())
    }
}
