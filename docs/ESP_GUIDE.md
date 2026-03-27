# ESP Data Extraction Guide

To implement the Bot-B-Gone Framework, you cannot use aggregated dashboard exports. You need a log of every single open and click event, with timestamps, user agents, and IP addresses. 

Here is how to get the raw data you need from the major Email Service Providers (ESPs).

## Sailthru
Sailthru provides excellent raw data access, which is why the reference implementation in this repo is built for it.
* **What to ask for:** Access to the Data Exporter or Jobs API.
* **Tables needed:** `sailthru_campaign_click` and `sailthru_campaign_open`.
* **Key fields:** `timestamp`, `user_agent`, `ip`, `url` (for clicks), and their proprietary `is_nhi` (Non-Human Interaction) flag.

## SendGrid
SendGrid is a pure infrastructure play, so you have to build the data pipeline yourself.
* **What to ask for:** Set up the **Event Webhook**.
* **How to configure:** Do not rely on the dashboard. Configure the webhook to POST every `open` and `click` event to your own database, AWS S3 bucket, or a service like Snowplow.
* **Key fields:** The webhook payload includes the `useragent`, `ip`, and `timestamp` required for the model.

## Mailchimp
Mailchimp's standard campaign reports are pre-aggregated and useless for bot filtering.
* **What to ask for:** You need the **Marketing API's Account Exports** feature or the Transactional API's `export-activity-history` endpoint.
* **How to configure:** Pull the raw activity CSVs on a scheduled basis and load them into your data warehouse.

## beehiiv
beehiiv's 3D Analytics dashboard is beautiful, but it is pre-filtered. You need the raw feed.
* **What to ask for:** Access to their **Enterprise API** or Webhooks.
* **How to configure:** Pull raw subscriber event logs. If you are on a plan that doesn't expose raw events, you will need to upgrade or negotiate access.

## Omeda
Omeda's data model is highly publisher-friendly, but you still need to bypass the dashboard.
* **What to ask for:** Use their **Email API**.
* **How to configure:** Specifically use the `Email Clicks` and `Email Deployment` services to pull raw behavioral actions tied to specific deployment IDs.

## ActiveCampaign
* **What to ask for:** Set up **Event Tracking** or use their **Webhooks**.
* **How to configure:** Push `Click` and `Open` events to your server. Their standard API limits historical event extraction, so real-time webhooks are the best path for building a data warehouse.

## HubSpot
* **What to ask for:** Access to the **Email Events API**.
* **How to configure:** Use the `GET /email/public/v1/events` endpoint to pull raw event data. Note that HubSpot has its own aggressive bot filtering; you want the raw, unfiltered event stream to run your own model.

## Klaviyo
* **What to ask for:** Use the **Metrics API**.
* **How to configure:** Query the `Get Metric Export` endpoint for the "Opened Email" and "Clicked Email" metrics. You need the raw event timeline, not the aggregated profiles.

## Campaign Monitor
* **What to ask for:** Set up their **Webhooks**.
* **How to configure:** Push `Click` and `Open` events to your server in real-time, or use their API to pull subscriber-level activity per campaign.

## ConvertKit (Kit)
* **What to ask for:** Use their **API or Webhooks**.
* **How to configure:** Pull raw subscriber engagement data. Their standard reporting is campaign-level; you need event-level data to run the Bot-B-Gone model.

## Constant Contact
* **What to ask for:** Use the **V3 API**.
* **How to configure:** Use the `Email Tracking` endpoints to pull detailed open and click activity reports.

## Marigold Engage (formerly Selligent)
* **What to ask for:** Access to the **Data Export** features or **API**.
* **How to configure:** As an enterprise platform, Marigold allows for raw data dumps. You need the granular interaction logs including user agents and IPs.

## Substack & Ghost
* **The Reality:** These platforms are designed for simplicity, not deep data engineering. 
* **Substack:** Does not currently provide API access to raw event logs (user agents, IPs, timestamps). You are entirely reliant on their internal bot filtering.
* **Ghost:** Provides webhooks, but the native email analytics are basic. You may need to integrate a third-party analytics tool or use Mailgun (their default sender) webhooks directly to get the raw data needed for this framework.

---
**The Golden Rule:** If your ESP will not give you individual event-level data with timestamps and user agents, you cannot build an independent model. You are entirely at their mercy.
