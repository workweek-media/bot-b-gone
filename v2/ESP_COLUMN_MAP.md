# ESP Column Mapping

Bot-B-Gone V2 expects these column names. Map your ESP's columns to ours.

## Required Columns

| Bot-B-Gone Column | Description |
|---|---|
| `subscriber_id` | Unique subscriber identifier |
| `blast_id` | Campaign / blast / send identifier |
| `send_timestamp` | When the email was dispatched |
| `open_timestamp` | When the subscriber opened (null if no open) |
| `click_timestamp` | When the subscriber clicked (null if no click) |

## Recommended Columns

| Bot-B-Gone Column | Description |
|---|---|
| `device` | Device or email client name |
| `click_urls` | Space-separated URLs clicked |
| `click_timestamps` | Pipe-separated per-click timestamps |
| `user_agent` | Full user agent string |

## ESP Mapping Reference

### Sailthru (Marigold Engage)

```python
column_map = {
    'profile_id': 'subscriber_id',
    'blast_id': 'blast_id',
    'send_time': 'send_timestamp',
    'open_time': 'open_timestamp',
    'click_time': 'click_timestamp',
    'device': 'device',
    'first_ten_clicks': 'click_urls',        # space-separated
    'first_ten_clicks_time': 'click_timestamps',  # pipe-separated
}
```

### Mailchimp

```python
# Use the Export API or Audience Activity endpoint
column_map = {
    'email_id': 'subscriber_id',    # or member.id
    'campaign_id': 'blast_id',
    'send_time': 'send_timestamp',  # from campaign settings
    'timestamp': 'open_timestamp',  # from activity with action='open'
    'timestamp': 'click_timestamp', # from activity with action='click'
    'url': 'click_urls',
    # device: parse from user_agent in activity data
}
```

### SendGrid

```python
# Use Event Webhook data
column_map = {
    'sg_message_id': 'subscriber_id',  # or email field
    'marketing_campaign_id': 'blast_id',
    'timestamp': 'send_timestamp',      # event='delivered'
    'timestamp': 'open_timestamp',      # event='open'
    'timestamp': 'click_timestamp',     # event='click'
    'url': 'click_urls',
    'useragent': 'user_agent',
}
```

### HubSpot

```python
# Use Email Events API
column_map = {
    'recipient': 'subscriber_id',
    'emailCampaignId': 'blast_id',
    'created': 'send_timestamp',     # event type='SENT'
    'created': 'open_timestamp',     # event type='OPEN'
    'created': 'click_timestamp',    # event type='CLICK'
    'url': 'click_urls',
    'userAgent': 'user_agent',
    # Note: HubSpot already filters some bot clicks
}
```

### Klaviyo

```python
# Use Metrics API (events endpoint)
column_map = {
    '$email': 'subscriber_id',       # or person_id
    'campaign_id': 'blast_id',
    'datetime': 'send_timestamp',    # from 'Received Email' metric
    'datetime': 'open_timestamp',    # from 'Opened Email' metric
    'datetime': 'click_timestamp',   # from 'Clicked Email' metric
    'URL': 'click_urls',
    'Client Name': 'device',
}
```

### Braze

```python
# Use Currents data export
column_map = {
    'external_user_id': 'subscriber_id',
    'campaign_id': 'blast_id',
    'time': 'send_timestamp',        # from email.Send events
    'time': 'open_timestamp',        # from email.Open events
    'time': 'click_timestamp',       # from email.Click events
    'url': 'click_urls',
    'user_agent': 'user_agent',
}
```

### Iterable

```python
# Use Data Export API
column_map = {
    'userId': 'subscriber_id',       # or email
    'campaignId': 'blast_id',
    'createdAt': 'send_timestamp',   # from emailSend events
    'createdAt': 'open_timestamp',   # from emailOpen events
    'createdAt': 'click_timestamp',  # from emailClick events
    'url': 'click_urls',
    # device: parse from metadata
}
```

### beehiiv

```python
# Use Analytics API or data export
column_map = {
    'subscriber_id': 'subscriber_id',
    'post_id': 'blast_id',
    'sent_at': 'send_timestamp',
    'opened_at': 'open_timestamp',
    'clicked_at': 'click_timestamp',
    'url': 'click_urls',
}
```

### ConvertKit (Kit)

```python
# Use API v3 subscribers + broadcasts
column_map = {
    'subscriber.id': 'subscriber_id',
    'broadcast.id': 'blast_id',
    'created_at': 'send_timestamp',
    # Open/click timestamps from webhook events
}
```

## Data Format Notes

- **One row per subscriber per blast.** If your ESP gives you one row per event, aggregate to subscriber-blast level first.
- **Timestamps** should be in ISO 8601 or any format pandas can parse.
- **click_urls** should be space-separated. If your ESP gives arrays or JSON, convert to space-separated.
- **Null handling**: Use empty/null for subscribers who didn't open or click. Don't exclude them — the model needs send counts.

## Minimum Data Volume

- **For subscriber profiling**: 90+ days of send data recommended (30 day minimum)
- **For ML training**: 50K+ subscriber-send rows recommended
- **For production classification**: Works on any blast size
