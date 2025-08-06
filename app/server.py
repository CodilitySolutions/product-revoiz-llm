import json
import os
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from concurrent.futures import TimeoutError as ConnectionTimeoutError
from retell import Retell
from .custom_types import (
    ConfigResponse,
    ResponseRequiredRequest,
)
# from .llm import LlmClient  # or use .llm_with_func_calling
from .llm_with_func_calling import LlmClient
# from .gemini_with_func_calling import LlmClient

load_dotenv(override=True)
app = FastAPI()
retell = Retell(api_key=os.environ["RETELL_API_KEY"])


# Handle webhook from Retell server. This is used to receive events from Retell server.
# Including call_started, call_ended, call_analyzed
@app.post("/webhook")
async def handle_webhook(request: Request):
    try:
        post_data = await request.json()
        
        # Validate that post_data contains required fields
        if not isinstance(post_data, dict):
            return JSONResponse(
                status_code=400,
                content={"message": "Invalid request format"}
            )
            
        if "event" not in post_data or "data" not in post_data:
            return JSONResponse(
                status_code=400,
                content={"message": "Missing required fields: event and data"}
            )

        # Verify signature
        valid_signature = retell.verify(
            json.dumps(post_data, separators=(",", ":"), ensure_ascii=False),
            api_key=str(os.environ["RETELL_API_KEY"]),
            signature=str(request.headers.get("X-Retell-Signature", "")),
        )
        
        if not valid_signature:
            print(
                "Received Unauthorized",
                post_data.get("event", "unknown"),
                post_data.get("data", {}).get("call_id", "unknown"),
            )
            return JSONResponse(status_code=401, content={"message": "Unauthorized"})

        # Handle different event types
        event_type = post_data["event"]
        call_id = post_data.get("data", {}).get("call_id", "unknown")
        
        if event_type == "call_started":
            print(f"Call started event: {call_id}")
        elif event_type == "call_ended":
            print(f"Call ended event: {call_id}")
        elif event_type == "call_analyzed":
            print(f"Call analyzed event: {call_id}")
        else:
            print(f"Unknown event type: {event_type} for call: {call_id}")
            
        return JSONResponse(status_code=200, content={"received": True})
        
    except json.JSONDecodeError:
        print("Error: Invalid JSON in webhook request")
        return JSONResponse(
            status_code=400,
            content={"message": "Invalid JSON format"}
        )
    except Exception as err:
        print(f"Error in webhook: {str(err)}")
        return JSONResponse(
            status_code=500,
            content={"message": f"Internal Server Error: {str(err)}"}
        )


# Start a websocket server to exchange text input and output with Retell server. Retell server
# will send over transcriptions and other information. This server here will be responsible for
# generating responses with LLM and send back to Retell server.
@app.websocket("/llm-websocket/{call_id}")
async def websocket_handler(websocket: WebSocket, call_id: str):
    try:
        await websocket.accept()
        llm_client = LlmClient()
        llm_client.setCallId(call_id)

        # Send optional config to Retell server
        config = ConfigResponse(
            response_type="config",
            config={
                "auto_reconnect": True,
                "call_details": True,
            },
            response_id=1,
        )
        await websocket.send_json(config.__dict__)

        # Send first message to signal ready of server
        response_id = 0
        first_event = llm_client.draft_begin_message()
        await websocket.send_json(first_event.__dict__)

        async def handle_message(request_json):
            nonlocal response_id

            try:
                # There are 5 types of interaction_type: call_details, pingpong, update_only, response_required, and reminder_required.
                # Not all of them need to be handled, only response_required and reminder_required.
                if request_json["interaction_type"] == "call_details":
                    print(json.dumps(request_json, indent=2))
                    return
                if request_json["interaction_type"] == "ping_pong":
                    await websocket.send_json(
                        {
                            "response_type": "ping_pong",
                            "timestamp": request_json["timestamp"],
                        }
                    )
                    return
                if request_json["interaction_type"] == "update_only":
                    return
                if (
                    request_json["interaction_type"] == "response_required"
                    or request_json["interaction_type"] == "reminder_required"
                ):
                    response_id = request_json["response_id"]
                    request = ResponseRequiredRequest(
                        interaction_type=request_json["interaction_type"],
                        response_id=response_id,
                        transcript=request_json["transcript"],
                    )
                    print(
                        f"""Received interaction_type={request_json['interaction_type']}, response_id={response_id}, last_transcript={request_json['transcript'][-1]['content']}"""
                    )

                    async for event in llm_client.draft_response(request):
                        await websocket.send_json(event.__dict__)
                        if request.response_id < response_id:
                            break  # new response needed, abandon this one
            except Exception as e:
                print(f"Error handling message: {str(e)}")
                # Try to send an error response to the client
                try:
                    await websocket.send_json({
                        "response_type": "response",
                        "response_id": response_id,
                        "content": "I apologize, but I encountered an error. Please try again.",
                        "content_complete": True,
                        "end_call": False
                    })
                except:
                    pass

        async for data in websocket.iter_json():
            asyncio.create_task(handle_message(data))

    except WebSocketDisconnect:
        print(f"LLM WebSocket disconnected for {call_id}")
    except ConnectionTimeoutError as e:
        print(f"Connection timeout error for {call_id}: {str(e)}")
    except Exception as e:
        print(f"Error in LLM WebSocket: {str(e)} for {call_id}")
        try:
            await websocket.close(1011, "Server error")
        except:
            pass
    finally:
        backend_api_url = llm_client.getBackendAPIUrl()
        order_details = llm_client.getCurrentOrder()
        await llm_client.saveOrder(backend_api_url, order_details)
        print(f"LLM WebSocket connection closed for {call_id}")
